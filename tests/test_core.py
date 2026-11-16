from __future__ import annotations
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import Counter
from app import csv_bytes
from engine import Engine, UserError, exam_allocation, exam_required
from importer import extract_bank, ImportFailure, render_cell, normalize
from tests.fixtures import make_pdf, install_fixture


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def test_regular_table_and_flags(self):
        q, report = extract_bank(make_pdf(self.root / 'test.pdf'), 'C')
        self.assertEqual(len(q), 3)
        self.assertEqual(q[0]['correct'], [2])
        self.assertEqual(q[1]['correct'], [0])
        self.assertEqual(q[2]['correct'], [0,2])
        self.assertEqual([x['section'] for x in q], [1,1,2])
        self.assertEqual(report['questionCount'], sum(p['count'] for p in report['pages']))
    def test_tiny_tip_column(self):
        q, report = extract_bank(make_pdf(self.root / 'test.pdf',tiny_column=True), 'C')
        self.assertEqual(len(q),3)
        self.assertEqual(q[0]['correct'],[2])
        self.assertNotIn('Corect',q[0]['text'])
    def test_trailing_empty_column_ignored(self):
        q, report = extract_bank(make_pdf(self.root / 'test.pdf', trailing_column=True), 'C')
        self.assertEqual(len(q), 3)
        self.assertEqual([x['correct'] for x in q], [[2], [0], [0, 2]])
        self.assertEqual(report['gradableCount'], 3)
    def test_sliver_with_bleeding_hyphen_ignored(self):
        q, report = extract_bank(make_pdf(self.root / 'test.pdf', sliver_bleed=True), 'C')
        self.assertEqual(len(q), 3)
        self.assertEqual([x['correct'] for x in q], [[2], [0], [0, 2]])
        self.assertEqual(report['gradableCount'], 3)
    def test_hyphen_bleeding_into_key_cell_keeps_key(self):
        q, report = extract_bank(make_pdf(self.root / 'test.pdf', key_bleed=True), 'C')
        self.assertEqual(report['gradableCount'], 3)
        self.assertEqual([x['correct'] for x in q], [[2], [0], [0, 2]])
        self.assertTrue(q[0]['options'][1].endswith('-'))
    def test_tip_digits_on_rule_removed_from_both_cells(self):
        q, report = extract_bank(make_pdf(self.root / 'test.pdf', tip_overlap=True), 'C')
        for x in q:
            self.assertNotIn('11', x['text'])
            self.assertNotIn('11', x['options'][0])
        self.assertEqual(q[0]['options'][0], 'Varianta de test A')
        self.assertTrue(q[0]['text'].endswith('aplicatiei?'))
    def test_incomplete_key_is_ungraded(self):
        q, report = extract_bank(make_pdf(self.root / 'test.pdf',invalid_key=True), 'C')
        self.assertFalse(q[-1]['gradable'])
        self.assertEqual(report['gradableCount'],2)
        self.assertTrue(report['warnings'])
    def test_vector_image_and_original_crop(self):
        path = make_pdf(self.root / 'test.pdf')
        q, report = extract_bank(path,'C')
        self.assertTrue(q[1]['visuals'][0])
        self.assertFalse(q[0]['visuals'][0])
        image = render_cell(path,q[1],0)
        self.assertTrue(image.startswith(b'\x89PNG'))
        self.assertGreater(len(image),500)
        # Crops are wholly inside the answer cell, excluding adjacent key cells.
        self.assertLess(q[0]['boxes'][1][2],q[0]['boxes'][2][0])
    def test_non_pdf_rejected(self):
        p=self.root/'error.pdf';p.write_text('<h1>429 Too Many Requests</h1>')
        with self.assertRaises(ImportFailure): extract_bank(p,'C')
    def test_no_table_rejected(self):
        import pymupdf as fitz
        p=self.root/'empty.pdf';d=fitz.open();d.new_page().insert_text((20,30),'Not a question table');d.save(p);d.close()
        with self.assertRaises(ImportFailure): extract_bank(p,'C')
    def test_category_validation(self):
        with self.assertRaises(ImportFailure): extract_bank(self.root/'nothing','X')
    def test_romanian_normalization(self):
        self.assertEqual(normalize('  \u015farpe\n \u0163ar\u0103  '),'\u0219arpe \u021bar\u0103')


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.engine=Engine(self.root/'date')
        self.q,self.report=install_fixture(self.engine,self.root)
        self.d,_=install_fixture(self.engine,self.root,'D')
    def tearDown(self):
        self.engine.close();self.tmp.cleanup()
    def start(self,cat='C',topic='all'):
        return self.engine.start(cat,topic)
    def answer(self,v,selected=None):
        q=self.engine.question(v['question']['id'])
        return self.engine.answer(v['session']['id'],q['id'],q['correct'] if selected is None else selected)
    def complete(self,v):
        while True:
            v=self.answer(v)
            if v['completed']:return v
            v=self.engine.next_question(v['session']['id'])
    def test_first_choice_feedback_idempotent(self):
        v=self.start();v=self.answer(v,[0]);self.assertEqual(v['feedback']['correct'],0)
        v=self.answer(v,[2]);self.assertEqual(v['feedback']['selected'],[0])
        self.assertEqual(v['run']['answered'],1);self.assertEqual(v['run']['wrong'],1)
    def test_keys_hidden_until_answer(self):
        v=self.start();self.assertNotIn('correct',v['question']);self.assertIsNone(v['feedback'])
        v=self.answer(v);self.assertEqual(v['feedback']['correctIndices'],[2])
    def test_next_requires_answer(self):
        v=self.start()
        with self.assertRaises(UserError):self.engine.next_question(v['session']['id'])
    def test_resume_same_question_and_confirmation(self):
        v=self.answer(self.start());rid=v['run']['id']
        self.engine.end(v['session']['id'],'closed',True)
        resumed=self.start();self.assertEqual(resumed['run']['id'],rid)
        self.assertEqual(resumed['question']['id'],v['question']['id'])
        self.assertEqual(resumed['feedback']['correct'],1)
        self.assertNotEqual(resumed['session']['id'],v['session']['id'])
        self.assertEqual(resumed['session']['answered'],0)
    def test_database_restart_persists_cursor_and_stats(self):
        v=self.answer(self.start());v=self.engine.next_question(v['session']['id']);qid=v['question']['id']
        self.engine.close();self.engine=Engine(self.root/'date')
        resumed=self.start();self.assertEqual(resumed['question']['id'],qid)
        self.assertEqual(resumed['run']['answered'],1)
        self.assertTrue(list((self.root/'date'/'backups').glob('*.sqlite3')))
    def test_categories_independent(self):
        c=self.answer(self.start());c=self.engine.next_question(c['session']['id'])
        d=self.answer(self.start('D'));back=self.start('C')
        self.assertEqual(back['question']['id'],c['question']['id'])
        self.assertEqual(back['run']['answered'],1)
        self.assertEqual(self.start('D')['feedback']['correct'],1)
    def test_reset_archives_and_preserves_stats(self):
        v=self.answer(self.start(),[0]);rid=v['run']['id']
        new=self.engine.reset(v['session']['id'])
        self.assertNotEqual(new['run']['id'],rid);self.assertEqual(new['run']['cursor'],0)
        stats=self.engine.statistics();old=next(r for r in stats['runs'] if r['id']==rid)
        self.assertEqual(old['status'],'reset');self.assertEqual(old['wrong'],1)
    def test_full_completion_counted_once(self):
        v=self.complete(self.start());self.assertTrue(v['completed'])
        self.answer(v);self.engine.next_question(v['session']['id'])
        s=self.engine.statistics();self.assertEqual(s['categories'][0]['completedRuns'],1)
        self.assertEqual(v['run']['percentage'],100)
        self.assertEqual(v['run']['answered'],3)
    def test_topic_completion_not_full_category(self):
        v=self.complete(self.start(topic=self.q[0]['topic']))
        self.assertTrue(v['completed']);self.assertFalse(v['run']['completeCategory'])
        self.assertEqual(self.engine.statistics()['categories'][0]['completedRuns'],0)
    def test_multi_answer_exact_set_required(self):
        v=self.answer(self.start());v=self.engine.next_question(v['session']['id']);v=self.answer(v);v=self.engine.next_question(v['session']['id'])
        self.assertTrue(v['question']['multiple']);v=self.answer(v,[0]);self.assertEqual(v['feedback']['correct'],0)
    def test_ungraded_not_in_percentage(self):
        q,_=install_fixture(self.engine,self.root,invalid_key=True)
        v=self.start();v=self.answer(v);v=self.engine.next_question(v['session']['id']);v=self.answer(v);v=self.engine.next_question(v['session']['id']);v=self.answer(v,[])
        self.assertTrue(v['completed']);self.assertEqual(v['run']['ungraded'],1)
        self.assertEqual(v['run']['wrong'],0);self.assertEqual(v['run']['percentage'],100)
    def test_invalid_selection_rejected(self):
        v=self.start()
        for selected in ([3],[True],[0,0],[0,1],[]):
            with self.assertRaises(UserError):self.answer(v,selected)
    def test_old_session_cannot_answer(self):
        old=self.start();self.start('D')
        with self.assertRaises(UserError):self.answer(old)
    def test_reset_category_archives_all_tracks_keeps_other_category(self):
        v=self.answer(self.start(),[0]);rid=v['run']['id']
        t=self.answer(self.start(topic=self.q[0]['topic']));tid=t['run']['id']
        d=self.answer(self.start('D'));did=d['run']['id']
        result=self.engine.reset_category('C')
        self.assertEqual(result['archivedRuns'],2)
        runs={r['id']:r for r in self.engine.statistics()['runs']}
        self.assertEqual(runs[rid]['status'],'reset');self.assertEqual(runs[tid]['status'],'reset')
        self.assertEqual(runs[rid]['wrong'],1)
        self.assertTrue(self.engine.state()['autoResume'])  # the last run was D, still active
        fresh=self.start();self.assertNotEqual(fresh['run']['id'],rid);self.assertEqual(fresh['run']['cursor'],0);self.assertIsNone(fresh['feedback'])
        self.assertEqual(self.start('D')['run']['id'],did)
        self.engine.reset_category('D');self.assertFalse(self.engine.state()['autoResume'])
    def test_delete_history_wipes_category_stats_keeps_other_and_backs_up(self):
        v=self.answer(self.start(),[0]);t=self.answer(self.start(topic=self.q[0]['topic']));d=self.answer(self.start('D'));did=d['run']['id']
        for f in (self.root/'date'/'backups').glob('*.sqlite3'):f.unlink()
        result=self.engine.delete_history('C');self.assertEqual(result['deletedRuns'],2)
        s=self.engine.statistics();c=next(x for x in s['categories'] if x['category']=='C');dd=next(x for x in s['categories'] if x['category']=='D')
        self.assertEqual((c['answered'],c['correct'],c['wrong'],c['completedRuns']),(0,0,0,0))
        self.assertEqual(dd['answered'],1);self.assertEqual([r['category'] for r in s['runs']],['D']);self.assertEqual([x['category'] for x in s['sessions']],['D'])
        self.assertTrue(list((self.root/'date'/'backups').glob('*.sqlite3')))
        fresh=self.start();self.assertEqual(fresh['run']['cursor'],0);self.assertIsNone(fresh['feedback'])
        self.assertEqual(self.start('D')['run']['id'],did)
        self.assertEqual(self.engine.delete_history('all')['deletedRuns'],2)
        self.assertEqual(self.engine.statistics()['runs'],[]);self.assertFalse(self.engine.state()['autoResume'])
        with self.assertRaises(UserError):self.engine.delete_history('X')
    def test_force_reimport_same_digest_restarts_from_zero(self):
        v=self.answer(self.start());rid=v['run']['id'];v=self.engine.next_question(v['session']['id'])
        bank=self.engine.bank_info('C')
        result=self.engine.install_bank('C',self.q,self.report,bank['source_file'],bank['original_url'],'manual',force=True)
        self.assertFalse(result['unchanged'])
        self.assertEqual(self.engine.bank_info('C')['digest'],bank['digest'])
        fresh=self.start();self.assertNotEqual(fresh['run']['id'],rid);self.assertEqual(fresh['run']['cursor'],0)
        self.assertEqual(fresh['question']['id'],self.q[0]['id'])
        old=next(r for r in self.engine.statistics()['runs'] if r['id']==rid);self.assertEqual(old['status'],'replaced');self.assertEqual(old['correct'],1)
    def test_reimport_same_digest_no_reset(self):
        v=self.answer(self.start());install_fixture(self.engine,self.root)
        # A newly generated PDF has new document metadata; compare the exact same bank.
        info=self.engine.bank_info('C')
        current=self.start();report=json.loads(info['report'])
        bank_q=[json.loads(r[0]) for r in self.engine.db.execute('SELECT payload FROM questions WHERE category=?',('C',))]
        self.engine.install_bank('C',bank_q,report,info['source_file'],info['original_url'],info['origin'])
        self.assertEqual(self.engine.start('C')['run']['id'],current['run']['id'])
    def test_new_bank_archives_history_keeps_snapshot(self):
        v=self.answer(self.start());old_id=v['question']['id'];rid=v['run']['id']
        install_fixture(self.engine,self.root,invalid_key=True)
        s=self.engine.statistics();old=next(r for r in s['runs'] if r['id']==rid)
        self.assertEqual(old['status'],'replaced');self.assertEqual(old['correct'],1)
        self.assertEqual(self.engine.question(old_id)['text'],self.q[0]['text'])
    def test_backup_integrity_and_sources(self):
        self.answer(self.start());raw=self.engine.backup_zip()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            self.assertIn('date/progres.sqlite3',z.namelist())
            self.assertEqual(len([n for n in z.namelist() if n.endswith('.pdf')]),2)
            p=self.root/'restored.sqlite3';p.write_bytes(z.read('date/progres.sqlite3'))
        db=sqlite3.connect(p);self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0],'ok')
        self.assertEqual(db.execute('SELECT COUNT(*) FROM answers').fetchone()[0],1);db.close()
    def test_foreground_timer(self):
        v=self.start();sid=v['session']['id']
        self.engine.db.execute('UPDATE sessions SET last_seen=100,active_seconds=0 WHERE id=?',(sid,));self.engine.db.commit()
        with patch('engine.time.time',return_value=112):self.engine.heartbeat(sid,False)
        with patch('engine.time.time',return_value=180):self.engine.heartbeat(sid,False)
        row=self.engine.db.execute('SELECT active_seconds FROM sessions WHERE id=?',(sid,)).fetchone()
        self.assertEqual(row[0],12)
    def test_category_validation(self):
        with self.assertRaises(UserError):self.engine.start('E')



class ExamAllocationTests(unittest.TestCase):
    def test_proportional_with_every_topic(self):
        counts={'Balizaj':25,'Constructie':157,'Legislatie':25,'Lumini':73,'Meteo':8,'Navigatie':39,'Notiuni':57,'Reguli':44,'Sonore':20,'Siguranta':68}
        plan=exam_allocation(counts);total=sum(counts.values())
        self.assertEqual(sum(plan.values()),26);self.assertEqual(set(plan),set(counts))
        self.assertEqual((plan['Meteo'],plan['Constructie'],plan['Lumini']),(1,8,4))
        for topic,n in counts.items():self.assertLess(abs(plan[topic]-26*n/total),1)
    def test_never_more_than_a_topic_holds(self):
        plan=exam_allocation({'a':1,'b':2,'c':100})
        self.assertEqual(plan,{'a':1,'b':1,'c':24})
        self.assertEqual(exam_allocation({'a':1,'b':40},30),{'a':1,'b':29})
    def test_bank_smaller_than_exam(self):
        self.assertEqual(exam_allocation({'a':2,'b':1,'empty':0}),{'a':2,'b':1})
        self.assertEqual(exam_allocation({}),{})
        self.assertEqual((exam_required(26),exam_required(3),exam_required(13)),(22,3,11))
    def test_more_topics_than_questions(self):
        plan=exam_allocation({f't{i:02d}':i+1 for i in range(30)})
        self.assertEqual(sum(plan.values()),26);self.assertEqual(set(plan.values()),{1})
        self.assertNotIn('t00',plan)  # the smallest topics are the ones left out


class ExamTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.engine=Engine(self.root/'date')
        self.q,self.report=install_fixture(self.engine,self.root)
    def tearDown(self):
        self.engine.close();self.tmp.cleanup()
    def bank(self,sizes=(('T1',40),('T2',15),('T3',5)),ungradable=()):
        """A synthetic bank large enough for a full exam; every key is [2]."""
        questions=[]
        for topic,n in sizes:
            for _ in range(n):
                i=len(questions)+1
                questions.append({**self.q[0],'id':f'C-{"0"*10}-{i:04d}','ordinal':i,'topic':topic,'gradable':topic not in ungradable})
        report={**self.report,'sha256':'f'*64,'questionCount':len(questions),'gradableCount':sum(q['gradable'] for q in questions)}
        self.engine.install_bank('C',questions,report,'C-TEST.pdf','https://example.invalid/TECHNICAL_TEST_ONLY.pdf','manual')
    def sit(self,wrong=0):
        """Answer a whole exam, the first `wrong` questions incorrectly."""
        v=self.engine.start('C','exam');seen=[]
        while True:
            seen.append(self.engine.question(v['question']['id']))
            v=self.engine.answer(v['session']['id'],v['question']['id'],[0] if len(seen)<=wrong else seen[-1]['correct'])
            if v['completed']:return v,seen
            v=self.engine.next_question(v['session']['id'])
    def test_draw_is_proportional_from_every_topic(self):
        self.bank()
        exam=self.engine.state()['categories'][0]['exam']
        self.assertEqual(exam,{'total':26,'required':22,'minutes':30,'topics':[{'topic':'T1','count':17},{'topic':'T2','count':7},{'topic':'T3','count':2}]})
        v,seen=self.sit()
        self.assertEqual(v['run']['total'],26);self.assertEqual(len({q['id'] for q in seen}),26)
        self.assertEqual(Counter(q['topic'] for q in seen),{'T1':17,'T2':7,'T3':2})
    def test_passed_with_exactly_the_minimum(self):
        self.bank();v,_=self.sit(wrong=4);r=v['run']
        self.assertEqual((r['correct'],r['wrong'],r['examRequired'],r['examPassed']),(22,4,22,True))
        self.assertTrue(r['exam']);self.assertFalse(r['completeCategory'])
        c=self.engine.statistics()['categories'][0]
        self.assertEqual((c['exams'],c['examsPassed'],c['completedRuns']),(1,1,0))
        header,row=csv_bytes(self.engine.statistics()['runs'],'runs').decode('utf-8-sig').splitlines()[:2]
        self.assertTrue(header.endswith('Simulare examen admisa'));self.assertTrue(row.endswith('True'))
    def test_failed_below_the_minimum(self):
        self.bank();v,_=self.sit(wrong=5)
        self.assertEqual((v['run']['correct'],v['run']['examPassed']),(21,False))
        c=self.engine.statistics()['categories'][0];self.assertEqual((c['exams'],c['examsPassed']),(1,0))
    def test_resume_keeps_the_draw_and_a_new_exam_starts_over(self):
        self.bank();v=self.engine.start('C','exam');rid=v['run']['id'];qid=v['question']['id']
        self.assertIsNone(v['run']['examPassed'])
        self.engine.end(v['session']['id'],'closed',True)
        self.assertEqual(self.engine.state()['lastTopic'],'exam')
        again=self.engine.start('C','exam');self.assertEqual((again['run']['id'],again['question']['id']),(rid,qid))
        self.assertEqual(self.engine.start('C','all')['run']['total'],60)  # other tracks are untouched
        fresh=self.engine.reset(self.engine.start('C','exam')['session']['id'])
        self.assertNotEqual(fresh['run']['id'],rid);self.assertEqual((fresh['run']['cursor'],fresh['run']['total']),(0,26))
        done,_=self.sit();self.assertEqual(done['run']['id'],fresh['run']['id'])
        self.assertNotEqual(self.engine.start('C','exam')['run']['id'],fresh['run']['id'])
    def clock(self,sid,seconds,last_seen=None):
        """Set the active time already spent by a session (the exam clock)."""
        self.engine.db.execute('UPDATE sessions SET active_seconds=?,last_seen=COALESCE(?,last_seen) WHERE id=?',(seconds,last_seen,sid));self.engine.db.commit()
    def test_clock_counts_active_time_across_sessions(self):
        self.bank();v=self.engine.start('C','exam');sid=v['session']['id']
        self.assertAlmostEqual(v['run']['examRemaining'],1800,delta=2)
        self.clock(sid,600,last_seen=1000)
        with patch('engine.time.time',return_value=1012):
            self.assertEqual(self.engine.heartbeat(sid,False)['examRemaining'],1188)  # 12 s in the foreground
        with patch('engine.time.time',return_value=1500):
            self.assertEqual(self.engine.heartbeat(sid,True)['examRemaining'],1188)  # hidden: the clock stood still
        with patch('engine.time.time',return_value=1700):
            self.assertEqual(self.engine.view(sid)['run']['examRemaining'],1158)  # no beacon: at most 30 s
        self.engine.end(sid,'paused')
        self.assertLessEqual(self.engine.state()['categories'][0]['tracks'][0]['examRemaining'],1188)
        again=self.engine.start('C','exam');self.assertEqual(again['run']['id'],v['run']['id'])
        self.assertAlmostEqual(again['run']['examRemaining'],1173,delta=16)  # paused time is not counted
        other=self.engine.start('C','all');self.assertIsNone(other['run'].get('examRemaining'))
        self.assertNotIn('examRemaining',self.engine.heartbeat(other['session']['id'],True))
    def test_timeout_ends_exam_and_unanswered_are_not_correct(self):
        self.bank();v=self.engine.start('C','exam');sid=v['session']['id']
        for _ in range(3):
            v=self.engine.answer(sid,v['question']['id'],[2]);v=self.engine.next_question(sid)
        self.assertFalse(self.engine.expire(sid)['completed'])  # the browser clock ran ahead: nothing happens
        self.clock(sid,1800)
        v=self.engine.answer(sid,v['question']['id'],[2]);r=v['run']
        self.assertTrue(v['completed']);self.assertIsNone(v['feedback'])  # the late answer is not recorded
        self.assertEqual((r['answered'],r['correct'],r['total'],r['examTimedOut'],r['examPassed'],r['examRemaining']),(3,3,26,True,False,None))
        self.assertEqual(v['session']['status'],'completed');self.assertFalse(self.engine.state()['autoResume'])
        c=self.engine.statistics()['categories'][0];self.assertEqual((c['exams'],c['examsPassed']),(1,0))
        self.assertNotEqual(self.engine.start('C','exam')['run']['id'],r['id'])
    def test_timeout_after_enough_correct_answers_is_still_passed(self):
        self.bank();v=self.engine.start('C','exam');sid=v['session']['id']
        for _ in range(22):
            v=self.engine.answer(sid,v['question']['id'],[2]);v=self.engine.next_question(sid)
        self.clock(sid,1795,last_seen=1000)
        with patch('engine.time.time',return_value=1004):self.assertFalse(self.engine.expire(sid)['completed'])
        with patch('engine.time.time',return_value=1006):r=self.engine.expire(sid)['run']
        self.assertEqual((r['status'],r['answered'],r['examTimedOut'],r['examPassed']),('completed',22,True,True))
    def test_exam_without_time_left_cannot_be_resumed_or_advanced(self):
        self.bank();v=self.engine.start('C','exam');sid=v['session']['id']
        v=self.engine.answer(sid,v['question']['id'],[2]);self.clock(sid,1800)
        v=self.engine.next_question(sid);self.assertTrue(v['completed']);self.assertEqual(v['run']['cursor'],0)
        v=self.engine.start('C','exam');sid=v['session']['id'];self.clock(sid,1800);self.engine.end(sid,'paused')
        v=self.engine.start('C','exam');self.assertTrue(v['completed']);self.assertTrue(v['run']['examTimedOut'])
        self.assertEqual(v['session']['status'],'completed')
    def test_finished_in_time_is_not_a_timeout(self):
        self.bank();v,_=self.sit();self.assertFalse(v['run']['examTimedOut']);self.assertIsNone(v['run']['examRemaining'])
        self.assertNotIn('examTimedOut',self.engine.view(self.engine.start('C','all')['session']['id'])['run'])
    def test_questions_without_a_complete_key_are_left_out(self):
        self.bank(ungradable=('T3',))
        self.assertEqual([t['topic'] for t in self.engine.state()['categories'][0]['exam']['topics']],['T1','T2'])
        v,seen=self.sit()
        self.assertEqual(v['run']['ungraded'],0);self.assertTrue(all(q['gradable'] for q in seen))
        self.assertEqual(Counter(q['topic'] for q in seen),{'T1':19,'T2':7})
    def test_bank_smaller_than_exam_uses_every_question(self):
        v,seen=self.sit()
        self.assertEqual((v['run']['total'],v['run']['examRequired'],v['run']['examPassed']),(3,3,True))
        self.assertIsNone(self.engine.state()['categories'][1]['exam'])  # D is not imported


if __name__=='__main__':
    unittest.main(verbosity=2)

"""Optional developer UI test. Requires Playwright + Chromium; not needed by users.
All PDFs are synthetic test fixtures, not an ANR study dataset.
"""
from __future__ import annotations
import argparse
import base64
import re
import urllib.request
import urllib.error
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests.fixtures import make_pdf
from playwright.sync_api import sync_playwright, expect
ROOT=Path(__file__).resolve().parents[1]


def run(output: Path, bridge: bool = False):
    output.mkdir(parents=True,exist_ok=True)
    checks=[];errors=[]
    def ok(label):
        checks.append(label);print('PASS:',label,flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)
        c=make_pdf(work/'C_TEST.pdf',tiny_column=True)
        d=make_pdf(work/'D_TEST.pdf',category='D')
        sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
        url=f'http://127.0.0.1:{port}'
        log=(output/'browser_server.log').open('w')
        proc=subprocess.Popen([sys.executable,str(ROOT/'app.py'),'--port',str(port),'--data-dir',str(work/'date'),'--no-browser','--no-download'],stdout=log,stderr=subprocess.STDOUT)
        try:
            time.sleep(.8)
            with sync_playwright() as p:
                executable=shutil.which('chromium') or None
                browser=p.chromium.launch(headless=True,executable_path=executable,args=['--no-sandbox'])
                context=browser.new_context(viewport={'width':1360,'height':1000},accept_downloads=True)
                def bootstrap(page):
                    page.on('pageerror',lambda e:errors.append(str(e)))
                    if not bridge:
                        page.goto(url)
                        return
                    # Render only local, generated HTML at about:blank. The browser
                    # has no network access in this mode. A test adapter calls the
                    # real local HTTP server through Python, with the normal token.
                    def local_request(arg):
                        body=base64.b64decode(arg['body64']) if arg.get('body64') else None
                        request=urllib.request.Request(url+arg['path'],data=body,method=arg.get('method','GET'),headers=arg.get('headers',{}))
                        try: response=urllib.request.urlopen(request,timeout=20)
                        except urllib.error.HTTPError as exc: response=exc
                        return {'status':response.status,'headers':dict(response.headers),'body64':base64.b64encode(response.read()).decode()}
                    html=urllib.request.urlopen(url).read().decode()
                    html=re.sub(r'<script src="/app.js" defer></script>','',html)
                    html=re.sub(r'<link rel="stylesheet" href="/style.css">','',html)
                    page.set_content(html)
                    page.add_style_tag(content=(ROOT/'static/style.css').read_text())
                    page.expose_function('localRequest',local_request)
                    page.add_script_tag(content=r"""
                    const bytes64=bytes=>{let s='';for(let i=0;i<bytes.length;i+=8192)s+=String.fromCharCode(...bytes.slice(i,i+8192));return btoa(s);};
                    window.fetch=async(path,init={})=>{
                        let body=init.body,bytes;
                        if(body instanceof Blob)bytes=new Uint8Array(await body.arrayBuffer());
                        else if(body!==undefined)bytes=new TextEncoder().encode(body);
                        const r=await window.localRequest({path:String(path),method:init.method||'GET',headers:init.headers||{},body64:bytes?bytes64(bytes):null});
                        return new Response(Uint8Array.from(atob(r.body64),c=>c.charCodeAt(0)),{status:r.status,headers:r.headers});
                    };
                    navigator.sendBeacon=(path,blob)=>{window.lastBeacon=blob.text().then(body=>fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body}));return true;};
                    const hydrate=()=>document.querySelectorAll('img[src^="/crop/"]').forEach(img=>{
                        const src=img.getAttribute('src');if(img.dataset.hydrating===src)return;img.dataset.hydrating=src;
                        fetch(src).then(r=>r.blob()).then(blob=>new Promise(resolve=>{const reader=new FileReader();reader.onload=()=>{img.src=reader.result;resolve();};reader.readAsDataURL(blob);}));
                    });
                    new MutationObserver(hydrate).observe(document.body,{subtree:true,childList:true});
                    window.testDownloads=[];
                    """)
                    page.add_script_tag(content=(ROOT/'static/app.js').read_text())
                    page.add_script_tag(content="saveBlob=async(blob,name)=>window.testDownloads.push({name,body64:bytes64(new Uint8Array(await blob.arrayBuffer()))});")
                def saved_download(page,button,path):
                    if bridge:
                        before=page.evaluate('window.testDownloads.length')
                        page.locator(button).click()
                        page.wait_for_function('(n)=>window.testDownloads.length>n',arg=before)
                        data=page.evaluate('window.testDownloads.at(-1)')
                        path.write_bytes(base64.b64decode(data['body64']))
                    else:
                        with page.expect_download() as down: page.locator(button).click()
                        down.value.save_as(path)
                page=context.new_page();bootstrap(page)
                expect(page.locator('#fileC')).to_be_attached()
                page.screenshot(path=str(output/'01_initial_setup.png'),full_page=True)
                page.locator('#fileC').set_input_files(c)
                expect(page.locator('[data-category="C"]')).to_contain_text('3 intrebari',timeout=15000)
                page.locator('#navSources').click();expect(page.locator('#fileD')).to_be_attached()
                page.locator('#fileD').set_input_files(d)
                expect(page.locator('.source-card').nth(1)).to_contain_text('2 intrebari',timeout=15000)
                ok('PDF upload/import from the UI, both C and D')
                page.locator('#navStudy').click();expect(page.locator('#startBtn')).to_be_enabled()
                page.screenshot(path=str(output/'02_home.png'),full_page=True)
                page.locator('#startBtn').click();expect(page.locator('#questionText')).to_contain_text('TEST TEHNIC')
                page.locator('[data-option="0"]').click();expect(page.locator('#feedback')).to_contain_text('Nu este corect.')
                expect(page.locator('#feedback')).to_contain_text('Varianta de test C')
                assert page.evaluate('model.run.wrong')==1
                page.screenshot(path=str(output/'03_wrong_feedback.png'),full_page=True)
                ok('Incorrect answer shows the correct full answer, with one saved attempt')
                sid=page.evaluate('model.session.id')
                if bridge:
                    page.evaluate('window.dispatchEvent(new Event("pagehide"))')
                    page.evaluate('window.lastBeacon')
                    page.close();page=context.new_page();bootstrap(page)
                else: page.reload()
                expect(page.locator('#feedback')).to_contain_text('Nu este corect.')
                assert page.evaluate('model.session.id')!=sid
                assert page.evaluate('model.run.answered')==1
                ok('Reload resumes the same confirmation, starts a new session, no duplicate attempt')
                page.locator('#nextBtn').click();expect(page.locator('#questionText')).to_contain_text('Identifica')
                expect(page.locator('.question-crop')).to_be_visible()
                page.wait_for_function('document.querySelector(".question-crop").naturalWidth>100')
                assert page.locator('.question-crop').evaluate('(img)=>img.naturalWidth')>100
                page.locator('[data-option="0"]').click();expect(page.locator('#feedback')).to_contain_text('Corect.')
                page.screenshot(path=str(output/'04_correct_with_figure.png'),full_page=True)
                ok('Original vector illustration rendered and correct answer confirmed')
                page.locator('#changeCategory').click();expect(page.locator('[data-category="D"]')).to_be_visible()
                page.locator('[data-category="D"]').click();page.locator('#startBtn').click()
                page.locator('[data-option="2"]').click();expect(page.locator('#feedback')).to_contain_text('Corect.')
                page.locator('#nextBtn').click();expect(page.locator('#questionText')).to_contain_text('Identifica')
                page.locator('#pauseBtn').click();expect(page.locator('#startBtn')).to_be_visible()
                page.locator('[data-category="C"]').click();page.locator('#startBtn').click()
                expect(page.locator('#feedback')).to_contain_text('Corect.')
                assert page.evaluate('model.run.cursor')==1
                ok('Switching categories and pausing preserves independent progress')
                page.locator('#nextBtn').click();expect(page.locator('#verifyBtn')).to_be_disabled()
                page.locator('[data-option="0"]').click();page.locator('[data-option="2"]').click()
                expect(page.locator('#verifyBtn')).to_be_enabled();page.locator('#verifyBtn').click()
                expect(page.locator('#feedback')).to_contain_text('Corect.')
                assert page.evaluate('model.completed') is True
                page.locator('#nextBtn').click();expect(page.locator('h1')).to_contain_text('cap-coada')
                page.screenshot(path=str(output/'05_completed.png'),full_page=True)
                ok('Multiple-correct-answer handling and category completion')
                page.locator('#navStats').click();expect(page.locator('.stats-overview')).to_contain_text('66,7%')
                page.locator('[data-tab="runs"]').click();expect(page.locator('tbody')).to_contain_text('Cap-coada')
                page.screenshot(path=str(output/'06_statistics.png'),full_page=True)
                page.locator('tbody tr').filter(has_text='Cap-coada').locator('[data-details]').click();expect(page.locator('#detailsDialog')).to_be_visible()
                page.locator('#filterMistakes').click();expect(page.locator('.detail-answer')).to_have_count(1)
                page.locator('#detailsClose').click()
                saved_download(page,'#csvSessions',output/'sessions_export.csv')
                assert 'Corecte' in (output/'sessions_export.csv').read_text(encoding='utf-8-sig')
                ok('Separate session/run statistics, review of mistakes, CSV export')
                page.locator('#navStudy').click()
                if page.locator('#otherBtn').count():page.locator('#otherBtn').click()
                else:page.locator('#changeCategory').click()
                page.locator('[data-category="C"]').click();page.locator('#startBtn').click()
                page.locator('[data-option="0"]').click();expect(page.locator('#feedback')).to_be_visible()
                page.locator('#resetBtn').click();expect(page.locator('#confirmDialog')).to_be_visible();page.locator('#dialogOk').click()
                expect(page.locator('#questionText')).to_contain_text('Ce varianta')
                assert page.evaluate('model.run.answered')==0
                page.locator('#navStats').click();page.locator('[data-tab="runs"]').click()
                expect(page.locator('tbody')).to_contain_text('Resetata')
                assert page.evaluate('stats.categories[0].completedRuns')==1
                ok('Reset starts from the beginning and retains completed and reset histories')
                # A new browser profile shares SQLite progress, not cookies.
                context2=browser.new_context(viewport={'width':390,'height':844},accept_downloads=True)
                page2=context2.new_page();bootstrap(page2);expect(page2.locator('#questionText')).to_contain_text('Ce varianta')
                assert page2.evaluate('model.run.answered')==0
                page2.locator('[data-option="2"]').click();expect(page2.locator('#feedback')).to_contain_text('Corect.')
                assert page2.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
                page2.screenshot(path=str(output/'07_mobile.png'),full_page=True)
                ok('Independent browser profile resumes local data; mobile layout has no horizontal overflow')
                page2.locator('#navSources').click()
                backup=output/'technical_backup.zip';saved_download(page2,'#backupBtn',backup)
                with zipfile.ZipFile(backup) as z:
                    assert 'date/progres.sqlite3' in z.namelist()
                    assert len([n for n in z.namelist() if n.endswith('.pdf')])==2
                ok('Full backup contains progress database and both imported PDFs')
                # Mutations must have the per-server token and local Origin/Host.
                result=context2.request.post(url+'/api/start',data={'category':'C'})
                assert result.status==403
                result=context2.request.get(url+'/api/state',headers={'Host':'evil.example'})
                assert result.status==400
                ok('Missing CSRF token and invalid Host are rejected')
                assert not errors,errors
                ok('No uncaught browser JavaScript errors')
                context2.close();context.close();browser.close()
        finally:
            proc.terminate();proc.wait(timeout=10);log.close()
    report={'status':'passed','browserMode':'local DOM with HTTP test adapter' if bridge else 'native localhost','checks':checks,'count':len(checks),'javascriptErrors':errors,
            'scope':'Synthetic PDF fixtures only. Official ANR PDFs were not available as local bytes.'}
    (output/'browser_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'test-output')
    parser.add_argument('--bridge',action='store_true')
    args=parser.parse_args();print(json.dumps(run(args.output,args.bridge),indent=2))

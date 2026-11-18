'use strict';
const $ = id => document.getElementById(id);
const main = $('main');
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtDate = t => t ? new Date(t * 1000).toLocaleString('ro-RO') : '-';
const fmtSize = b => b > 1048576 ? (b / 1048576).toFixed(1) + ' MB' : b > 1024 ? (b / 1024).toFixed(0) + ' KB' : b + ' B';
let state = { view: 'documents', kbs: [], kb: 'default', me: null };

function toast(msg, ms = 3500) { const t = $('toast'); t.textContent = msg; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => t.hidden = true, ms); }

async function api(path, opts = {}) {
  const headers = Object.assign({ 'X-Requested-With': 'rokid-admin' }, opts.headers || {});
  if (opts.json !== undefined) { headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(opts.json); }
  const r = await fetch(path, { method: opts.method || (opts.body ? 'POST' : 'GET'), headers, body: opts.body, credentials: 'same-origin' });
  if (r.status === 204) return null;
  let data = null; try { data = await r.json(); } catch { data = null; }
  if (!r.ok) { if (r.status === 401 && state.me) { state.me = null; renderLogin(); } throw new Error((data && (data.detail?.[0]?.msg || data.detail)) || ('HTTP ' + r.status)); }
  return data;
}

async function boot() {
  try { const h = await fetch('/api/v1/health').then(r => r.json()); $('version').textContent = ' v' + h.version; } catch {}
  try { state.me = await api('/api/v1/admin/me'); } catch { state.me = null; }
  if (!state.me) return renderLogin();
  await loadKbs(); setView(state.view);
}

function renderLogin() {
  $('nav').hidden = true;
  main.innerHTML = `<div class="card login"><h1>Autentificare administrator</h1><p class="muted small">Interfata este destinata exclusiv administrarii backend-ului. Dispozitivele Rokid folosesc tokenuri separate.</p>
  <form id="loginForm"><label for="u">Utilizator</label><input id="u" autocomplete="username" required><label for="p">Parola</label><input id="p" type="password" autocomplete="current-password" required>
  <div class="actions"><button class="btn" type="submit">Intra</button></div><div id="loginMsg"></div></form></div>`;
  $('loginForm').onsubmit = async e => { e.preventDefault(); try { await api('/api/v1/admin/login', { json: { username: $('u').value, password: $('p').value } }); state.me = { scope: 'admin' }; await loadKbs(); setView('documents'); } catch (err) { $('loginMsg').innerHTML = `<div class="notice error">${esc(err.message)}</div>`; } };
}

async function loadKbs() { state.kbs = await api('/api/v1/knowledge-bases'); if (!state.kbs.some(k => k.id === state.kb)) state.kb = state.kbs[0]?.id || 'default'; }
function kbSelect(id = 'kbSel') { return `<label for="${id}">Baza de cunostinte</label><select id="${id}">${state.kbs.map(k => `<option value="${esc(k.id)}" ${k.id === state.kb ? 'selected' : ''}>${esc(k.name)} (${esc(k.id)})</option>`).join('')}</select>`; }
function bindKb(id = 'kbSel', cb) { const s = $(id); if (s) s.onchange = () => { state.kb = s.value; cb && cb(); }; }

function setView(view) {
  state.view = view; $('nav').hidden = false;
  document.querySelectorAll('#nav [data-view]').forEach(b => b.classList.toggle('active', b.dataset.view === view));
  ({ documents: renderDocuments, qa: renderQa, kb: renderKb, test: renderTest, candidates: renderCandidates, status: renderStatus })[view]();
}
document.querySelectorAll('#nav [data-view]').forEach(b => b.onclick = () => setView(b.dataset.view));
$('logout').onclick = async () => { await api('/api/v1/admin/logout', { method: 'POST' }); state.me = null; renderLogin(); };

function agreementLabel(a) { return ({ verified: 'verificat', conflict: 'CONFLICT intre inregistrarile verificate', consensus: 'documentatie + internet de acord', docs_only: 'doar documentatie', web_only: 'doar internet', disagreement: 'DEZACORD documentatie / internet', none: '-' })[a] || a; }
function agreementClass(a) { return a === 'verified' || a === 'consensus' ? 'ok' : a === 'disagreement' || a === 'conflict' ? 'failed' : a === 'none' ? '' : 'pending'; }
function sourceLine(s) { const name = esc(s.document_name) + (s.page ? ' p.' + s.page : '') + (s.section ? ' · ' + esc(s.section) : ''); return `<li>${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${name}</a>` : name} <span class="tag">${esc(s.kind)}</span>${s.score != null ? ' <span class="muted small">' + s.score + '</span>' : ''}</li>`; }

function warnList(ws) { return ws && ws.length ? `<details><summary>${ws.length} observatii</summary><ul class="small">${ws.map(w => `<li>${esc(w)}</li>`).join('')}</ul></details>` : ''; }

async function renderDocuments() {
  const docs = await api('/api/v1/documents?knowledge_base_id=' + encodeURIComponent(state.kb));
  main.innerHTML = `<h1>Documente</h1><div class="card"><h2>Incarca document</h2><p class="muted small">PDF, DOCX, TXT, MD, CSV, JSON, XLSX. PDF-urile scanate trec prin OCR daca Tesseract este instalat pe server.</p>
  <div class="row">${kbSelect()}<div><label for="docFile">Fisier</label><input id="docFile" type="file" accept=".pdf,.docx,.txt,.md,.markdown,.csv,.json,.xlsx"></div><div><label for="replaceId">Inlocuieste documentul (optional, id)</label><input id="replaceId" placeholder="doc_..."></div></div>
  <div class="actions"><button class="btn" id="uploadDoc">Incarca si indexeaza</button><button class="btn secondary" id="refreshDocs">Actualizeaza</button></div><div id="docMsg"></div></div>
  <div class="card"><table><thead><tr><th>Fisier</th><th>Stare</th><th>Pagini / fragmente</th><th>Versiune</th><th>Indexat</th><th></th></tr></thead><tbody>
  ${docs.length ? docs.map(d => `<tr><td><strong>${esc(d.filename)}</strong><div class="small muted code">${esc(d.id)} · ${fmtSize(d.size_bytes)} · ${esc(d.embedding_provider || '')}</div>${d.error ? `<div class="notice error small">${esc(d.error)}</div>` : ''}${warnList(d.warnings)}</td>
  <td><span class="tag ${esc(d.status)}">${esc(d.status)}</span></td><td>${d.pages || '-'} / ${d.chunk_count}</td><td>v${d.version}</td><td class="small">${fmtDate(d.indexed_at)}</td>
  <td><div class="actions"><button class="btn small secondary" data-reindex="${esc(d.id)}">Reindexeaza</button><button class="btn small danger" data-del="${esc(d.id)}">Sterge</button></div></td></tr>`).join('') : '<tr><td colspan="6" class="muted">Niciun document in aceasta baza de cunostinte.</td></tr>'}</tbody></table></div>`;
  bindKb('kbSel', renderDocuments); $('refreshDocs').onclick = renderDocuments;
  $('uploadDoc').onclick = async () => { const f = $('docFile').files[0]; if (!f) return toast('Alege un fisier.'); const fd = new FormData(); fd.append('file', f); fd.append('knowledge_base_id', state.kb); if ($('replaceId').value.trim()) fd.append('replace_document_id', $('replaceId').value.trim());
    $('uploadDoc').disabled = true; try { const r = await api('/api/v1/documents', { method: 'POST', body: fd }); $('docMsg').innerHTML = `<div class="notice ${r.duplicate ? '' : 'good'}">${esc(r.message)}</div>`; if (r.job_id) pollJob(r.job_id, renderDocuments); else renderDocuments(); } catch (e) { $('docMsg').innerHTML = `<div class="notice error">${esc(e.message)}</div>`; } finally { $('uploadDoc').disabled = false; } };
  document.querySelectorAll('[data-reindex]').forEach(b => b.onclick = async () => { try { const r = await api(`/api/v1/documents/${b.dataset.reindex}/reindex`, { method: 'POST' }); toast(r.message); pollJob(r.job_id, renderDocuments); } catch (e) { toast(e.message); } });
  document.querySelectorAll('[data-del]').forEach(b => b.onclick = async () => { if (!confirm('Stergi documentul si fragmentele indexate?')) return; try { await api(`/api/v1/documents/${b.dataset.del}`, { method: 'DELETE' }); toast('Document sters.'); renderDocuments(); } catch (e) { toast(e.message); } });
}

async function pollJob(jobId, done) {
  for (let i = 0; i < 600; i++) {
    await new Promise(r => setTimeout(r, i < 10 ? 700 : 2000));
    try { const j = await api('/api/v1/admin/jobs/' + jobId); if (j.status === 'done' || j.status === 'failed') { toast((j.status === 'done' ? 'Gata: ' : 'Esuat: ') + j.message, 6000); return done && done(); } } catch { return; }
  }
}

async function renderQa() {
  const sets = await api('/api/v1/qa-datasets?knowledge_base_id=' + encodeURIComponent(state.kb));
  main.innerHTML = `<h1>Seturi de intrebari verificate</h1><div class="card"><h2>Incarca set Q&amp;A</h2><p class="muted small">JSON / JSONL / CSV cu campurile question, options, correct_answer, explanation; sau PDF-ul oficial ANR (alege categoria).</p>
  <div class="row">${kbSelect()}<div><label for="qaFile">Fisier</label><input id="qaFile" type="file" accept=".json,.jsonl,.csv,.pdf"></div><div><label for="qaName">Nume (optional)</label><input id="qaName"></div><div><label for="anrCat">Categorie ANR (doar PDF)</label><select id="anrCat"><option value="">-</option><option value="C">C</option><option value="D">D</option></select></div></div>
  <div class="actions"><button class="btn" id="uploadQa">Incarca si indexeaza</button><button class="btn secondary" id="syncCaa">Preia PDF-urile ANR din Antrenor CAA</button><button class="btn secondary" id="refreshQa">Actualizeaza</button></div><div id="qaMsg"></div></div>
  <div class="card"><table><thead><tr><th>Set</th><th>Stare</th><th>Inregistrari</th><th>Invalide</th><th>Creat</th><th></th></tr></thead><tbody>
  ${sets.length ? sets.map(d => `<tr><td><strong>${esc(d.name)}</strong><div class="small muted code">${esc(d.id)} · ${esc(d.filename)} · ${esc(d.format)}</div>${d.error ? `<div class="notice error small">${esc(d.error)}</div>` : ''}${warnList(d.warnings)}</td>
  <td><span class="tag ${esc(d.status)}">${esc(d.status)}</span></td><td>${d.record_count}</td><td>${d.invalid_count}</td><td class="small">${fmtDate(d.created_at)}</td>
  <td><div class="actions"><button class="btn small secondary" data-records="${esc(d.id)}">Inregistrari</button><button class="btn small secondary" data-reindex="${esc(d.id)}">Reindexeaza</button><button class="btn small danger" data-del="${esc(d.id)}">Sterge</button></div></td></tr>`).join('') : '<tr><td colspan="6" class="muted">Niciun set in aceasta baza de cunostinte.</td></tr>'}</tbody></table></div><div id="records"></div>`;
  bindKb('kbSel', renderQa); $('refreshQa').onclick = renderQa;
  $('uploadQa').onclick = async () => { const f = $('qaFile').files[0]; if (!f) return toast('Alege un fisier.'); const fd = new FormData(); fd.append('file', f); fd.append('knowledge_base_id', state.kb); fd.append('name', $('qaName').value); if ($('anrCat').value) fd.append('anr_category', $('anrCat').value);
    $('uploadQa').disabled = true; try { const r = await api('/api/v1/qa-datasets', { method: 'POST', body: fd }); $('qaMsg').innerHTML = `<div class="notice ${r.duplicate ? '' : 'good'}">${esc(r.message)}</div>`; if (r.job_id) pollJob(r.job_id, renderQa); else renderQa(); } catch (e) { $('qaMsg').innerHTML = `<div class="notice error">${esc(e.message)}</div>`; } finally { $('uploadQa').disabled = false; } };
  $('syncCaa').onclick = async () => { $('syncCaa').disabled = true; try { const fd = new FormData(); fd.append('knowledge_base_id', state.kb); const r = await api('/api/v1/qa-datasets/sync-caa', { method: 'POST', body: fd });
    $('qaMsg').innerHTML = `<div class="notice">${r.results.map(x => `Categoria ${esc(x.category)}: ${esc(x.message)}`).join('<br>')}</div>`; const jobs = r.results.filter(x => x.job_id); if (jobs.length) jobs.forEach(x => pollJob(x.job_id, renderQa)); else renderQa(); } catch (e) { $('qaMsg').innerHTML = `<div class="notice error">${esc(e.message)}</div>`; } finally { if ($('syncCaa')) $('syncCaa').disabled = false; } };
  document.querySelectorAll('[data-reindex]').forEach(b => b.onclick = async () => { try { const r = await api(`/api/v1/qa-datasets/${b.dataset.reindex}/reindex`, { method: 'POST' }); toast(r.message); pollJob(r.job_id, renderQa); } catch (e) { toast(e.message); } });
  document.querySelectorAll('[data-del]').forEach(b => b.onclick = async () => { if (!confirm('Stergi setul si inregistrarile indexate?')) return; try { await api(`/api/v1/qa-datasets/${b.dataset.del}`, { method: 'DELETE' }); toast('Set sters.'); renderQa(); } catch (e) { toast(e.message); } });
  document.querySelectorAll('[data-records]').forEach(b => b.onclick = async () => { const r = await api(`/api/v1/qa-datasets/${b.dataset.records}/records?limit=50`); $('records').innerHTML = `<div class="card"><h2>Primele ${r.records.length} din ${r.total} inregistrari</h2>${r.records.map(x => `<div class="small" style="margin-bottom:10px"><strong>${esc(x.question)}</strong><br>${x.options.map(o => `${esc(o.label)}. ${esc(o.text)}`).join(' · ')}<br><span class="tag answered">corect: ${esc(x.correct_labels.join(', ') || x.correct_texts.join('; '))}</span> ${x.explanation ? '<span class="muted">' + esc(x.explanation) + '</span>' : ''} <span class="muted">${esc(JSON.stringify(x.source))}</span></div>`).join('')}</div>`; });
}

async function renderKb() {
  await loadKbs();
  main.innerHTML = `<h1>Baze de cunostinte</h1><div class="card"><h2>Creeaza</h2><div class="row"><div><label for="kbId">Id (litere, cifre, - _)</label><input id="kbId" placeholder="python"></div><div><label for="kbName">Nume</label><input id="kbName"></div><div><label for="kbDesc">Descriere</label><input id="kbDesc"></div></div><div class="actions"><button class="btn" id="createKb">Creeaza</button></div><div id="kbMsg"></div></div>
  <div class="card"><table><thead><tr><th>Id</th><th>Nume</th><th>Documente</th><th>Inregistrari Q&amp;A</th><th>Creat</th><th></th></tr></thead><tbody>${state.kbs.map(k => `<tr><td class="code">${esc(k.id)}</td><td><strong>${esc(k.name)}</strong><div class="small muted">${esc(k.description)}</div></td><td>${k.document_count}</td><td>${k.qa_record_count}</td><td class="small">${fmtDate(k.created_at)}</td><td>${k.id === 'default' ? '' : `<button class="btn small danger" data-delkb="${esc(k.id)}">Sterge</button>`}</td></tr>`).join('')}</tbody></table></div>`;
  $('createKb').onclick = async () => { try { await api('/api/v1/knowledge-bases', { json: { id: $('kbId').value.trim(), name: $('kbName').value.trim(), description: $('kbDesc').value.trim() } }); toast('Baza creata.'); renderKb(); } catch (e) { $('kbMsg').innerHTML = `<div class="notice error">${esc(e.message)}</div>`; } };
  document.querySelectorAll('[data-delkb]').forEach(b => b.onclick = async () => { if (!confirm('Stergi baza de cunostinte cu toate documentele si seturile ei?')) return; try { await api(`/api/v1/knowledge-bases/${b.dataset.delkb}`, { method: 'DELETE' }); toast('Stearsa.'); renderKb(); } catch (e) { toast(e.message); } });
}

function renderTest() {
  main.innerHTML = `<h1>Test RAG</h1><div class="card"><div class="row">${kbSelect()}</div>
  <p class="muted small">Acelasi apel ca al ochelarilor: <span class="code">POST /api/v1/questions/answer</span> cu <span class="code">text</span> si/sau <span class="code">image</span>. Textul se incearca primul; poza este citita cu modelul configurat doar daca textul nu da raspuns.</p>
  <label for="qText">Blocul de text, exact cum vine de la ochelari: intrebarea urmata de variantele A. / B. / C. pe linii separate (sau doar intrebarea, pentru o intrebare deschisa)</label>
  <textarea id="qText" rows="8" placeholder="Tuturor navelor aflate în largul mării ori pe toate apele învecinate accesibile navelor maritime li se aplică prevederile:
A. COLREG-1972.
B. R.N.D.
C. Codului ISPS."></textarea>
  <label for="qImage" style="margin-top:12px">Poza cu intrebarea (PNG/JPEG/WEBP), optional</label>
  <div class="row"><div><input id="qImage" type="file" accept="image/png,image/jpeg,image/webp,image/gif"></div></div>
  <div class="actions"><button class="btn" id="ask">Trimite</button><button class="btn secondary" id="clearText">Goleste</button></div></div><div id="answer"></div>`;
  bindKb('kbSel');
  $('clearText').onclick = () => { $('qText').value = ''; $('qImage').value = ''; $('answer').innerHTML = ''; };
  $('ask').onclick = async () => { const text = $('qText').value.trim(); const file = $('qImage').files[0];
    if (!text && !file) return toast('Lipeste blocul cu intrebarea sau alege o poza.');
    const fd = new FormData(); fd.append('request_id', crypto.randomUUID()); fd.append('knowledge_base_id', state.kb); if (text) fd.append('text', text); if (file) fd.append('image', file);
    $('ask').disabled = true; $('answer').innerHTML = '<p class="muted small">Se cauta raspunsul...</p>';
    try { const r = await api('/api/v1/questions/answer?full=1', { method: 'POST', body: fd });
      const compact = r.status === 'answered' && r.answer ? JSON.stringify(Object.assign(r.answer.option_labels.length ? { label: r.answer.option_labels.join(',') } : {}, { text: r.answer.text }, r.agreement === 'disagreement' ? { note: 'verifica: documentatia si internetul nu sunt de acord' } : {})) : JSON.stringify({ error: r.message || r.status });
      $('answer').innerHTML = `<div class="card"><h2>Raspuns <span class="tag ${esc(r.status)}">${esc(r.status)}</span> <span class="tag">${esc(r.match_type)}</span> <span class="tag ${agreementClass(r.agreement)}">${esc(agreementLabel(r.agreement))}</span> <span class="muted small">${r.processing_time_ms} ms</span></h2>
      ${r.answer ? `<p><strong>${esc(r.answer.option_labels.join(', '))}</strong> ${esc(r.answer.text)}</p><p class="muted">${esc(r.answer.explanation)}</p>${r.message ? `<div class="notice">${esc(r.message)}</div>` : ''}` : `<div class="notice">${esc(r.message)}</div>`}
      ${r.candidate_id ? `<p class="muted small">Raspuns neverificat, pastrat pentru confirmare in <a href="#" id="goCand">Candidati</a> (<span class="code">${esc(r.candidate_id)}</span>).</p>` : ''}
      <p class="muted small">Ce primesc ochelarii (fara ?full=1): <span class="code">${esc(compact)}</span></p>
      ${r.transcribed_text ? `<h2>Text extras din poza</h2><pre>${esc(r.transcribed_text)}</pre>` : ''}
      <h2>Surse</h2>${r.sources.length ? `<ul>${r.sources.map(sourceLine).join('')}</ul>` : '<p class="muted">Nicio sursa.</p>'}
      <details><summary>Timpi pe etape si JSON complet</summary><pre>${esc(JSON.stringify(r, null, 2))}</pre></details></div>`; const go = $('goCand'); if (go) go.onclick = e => { e.preventDefault(); setView('candidates'); }; } catch (e) { $('answer').innerHTML = `<div class="notice error">${esc(e.message)}</div>`; } finally { $('ask').disabled = false; } };
}

async function renderCandidates() {
  const status = state.candStatus || 'pending';
  const rows = await api(`/api/v1/admin/candidates?knowledge_base_id=${encodeURIComponent(state.kb)}&status=${encodeURIComponent(status)}&limit=100`);
  const names = { pending: 'in asteptare', confirmed: 'confirmate', rejected: 'respinse', all: 'toate' };
  main.innerHTML = `<h1>Raspunsuri de confirmat</h1><div class="card"><p class="muted small">Raspunsurile date din documentatie sau de pe internet (nu din seturile verificate) ajung aici. Confirmarea le muta in setul verificat „Confirmate din admin”: data viitoare aceeasi intrebare primeste raspunsul verificat in cateva milisecunde. Verifica litera inainte de confirmare; o poti schimba din lista.</p>
  <div class="row">${kbSelect()}<div><label for="candStatus">Stare</label><select id="candStatus">${Object.keys(names).map(k => `<option value="${k}" ${k === status ? 'selected' : ''}>${names[k]}</option>`).join('')}</select></div></div></div>
  ${rows.length ? rows.map(c => `<div class="card"><strong>${esc(c.question)}</strong><div class="small" style="margin:6px 0">${c.options.map(o => `<div>${esc(o.label)}. ${esc(o.text)}</div>`).join('')}</div>
    <p><span class="tag answered">propus: ${esc(c.option_labels.join(', ') || c.answer_text)}</span> <span class="tag ${agreementClass(c.agreement)}">${esc(agreementLabel(c.agreement))}</span> <span class="tag">${esc(c.match_type)}</span> <span class="muted small">documentatie: ${esc(c.docs_labels.join(',') || '-')} · internet: ${esc(c.web_labels.join(',') || '-')} · vazut de ${c.seen_count} ori · ${fmtDate(c.updated_at)}</span></p>
    <p class="muted small">${esc(c.explanation)}</p>${c.sources.length ? `<ul class="small">${c.sources.map(sourceLine).join('')}</ul>` : ''}
    ${c.status === 'pending' ? `<div class="actions">${c.options.length ? `<select data-pick="${esc(c.id)}">${c.options.map(o => `<option value="${esc(o.label)}" ${c.option_labels.includes(o.label) ? 'selected' : ''}>${esc(o.label)}</option>`).join('')}</select>` : ''}<button class="btn" data-confirm="${esc(c.id)}">Confirma</button><button class="btn danger" data-reject="${esc(c.id)}">Respinge</button></div>` : `<p class="small"><span class="tag ${c.status === 'confirmed' ? 'ok' : 'failed'}">${esc(names[c.status] || c.status)}</span>${c.record_id ? ' <span class="code">' + esc(c.record_id) + '</span>' : ''}</p>`}
  </div>`).join('') : '<div class="card"><p class="muted">Niciun candidat.</p></div>'}`;
  bindKb('kbSel', renderCandidates); $('candStatus').onchange = () => { state.candStatus = $('candStatus').value; renderCandidates(); };
  document.querySelectorAll('[data-confirm]').forEach(b => b.onclick = async () => { const sel = document.querySelector(`[data-pick="${b.dataset.confirm}"]`); const labels = sel && sel.value ? [sel.value] : null; b.disabled = true; try { const r = await api(`/api/v1/admin/candidates/${b.dataset.confirm}/confirm`, { json: { option_labels: labels } }); toast(r.message); renderCandidates(); } catch (e) { toast(e.message); b.disabled = false; } });
  document.querySelectorAll('[data-reject]').forEach(b => b.onclick = async () => { try { await api(`/api/v1/admin/candidates/${b.dataset.reject}/reject`, { method: 'POST' }); toast('Respins.'); renderCandidates(); } catch (e) { toast(e.message); } });
}

async function renderStatus() {
  const [s, jobs, tokens, health] = await Promise.all([api('/api/v1/admin/status'), api('/api/v1/admin/jobs?limit=30'), api('/api/v1/admin/tokens'), fetch('/api/v1/health').then(r => r.json())]);
  main.innerHTML = `<h1>Stare sistem</h1><div class="card"><div class="kv"><span>Sanatate</span><span class="tag ${esc(health.status)}">${esc(health.status)}</span><span>Baza de date</span><span>${esc(health.components.database)} · ${esc(s.database_url || '')}${s.database_warning ? `<div class="notice error small">${esc(s.database_warning)}</div>` : ''}</span><span>Model LLM</span><span>${esc(s.llm_model || '-')}</span><span>Model vedere (poze)</span><span>${esc(s.vision_model || s.llm_model || '-')}</span><span>ChromaDB</span><span>${esc(health.components.vector_store)}</span><span>Furnizor LLM</span><span>${esc(s.llm_provider)}</span><span>Embeddings</span><span>${esc(s.embedding_provider)}</span><span>Intrebari necunoscute</span><span>${s.answer_grounding === 'open' ? 'documentatie + cunostintele modelului' : 'doar dovezile incarcate (strict)' + (s.general_knowledge_fallback ? ', cunostinte generale cand nu exista dovezi' : '')}${s.single_answer ? ' · un singur raspuns corect' : ''}</span><span>Cautare web</span><span>${s.web_search ? esc(s.web_search) + ' (in paralel cu documentatia)' : 'dezactivata'}</span><span>Candidati de confirmat</span><span>${s.candidates_pending}</span><span>Documente</span><span>${esc(JSON.stringify(s.documents))}</span><span>Seturi Q&amp;A</span><span>${esc(JSON.stringify(s.qa_datasets))} · ${s.qa_records} inregistrari</span><span>Joburi in lucru</span><span>${s.jobs_running}</span><span>Date</span><span class="code">${esc(s.data_dir)}</span></div>
  ${s.reindex_needed && (s.reindex_needed.documents.length || s.reindex_needed.qa_datasets.length) ? `<div class="notice error small" style="margin-top:10px">Indexate cu alt furnizor de embeddings decat cel curent; reindexeaza-le din Documente / Seturi Q&amp;A, altfel nu apar la potrivirea semantica: ${esc([...s.reindex_needed.documents, ...s.reindex_needed.qa_datasets].join(', '))}</div>` : ''}
  ${s.failed_documents.length || s.failed_datasets.length ? `<h2 style="margin-top:14px">Erori de ingestie</h2><ul class="small">${s.failed_documents.map(d => `<li><strong>${esc(d.filename)}</strong>: ${esc(d.error)}</li>`).join('')}${s.failed_datasets.map(d => `<li><strong>${esc(d.name)}</strong>: ${esc(d.error)}</li>`).join('')}</ul>` : ''}</div>
  <div class="card"><h2>Tokenuri API</h2><p class="muted small">Dispozitivul Rokid primeste un token cu scope <code>device</code> (poate doar sa intrebe). Tokenul este afisat o singura data.</p><div class="row"><div><label for="tokName">Nume</label><input id="tokName" placeholder="Rokid Glasses"></div><div><label for="tokScope">Scope</label><select id="tokScope"><option value="device">device</option><option value="admin">admin</option></select></div></div><div class="actions"><button class="btn" id="mkTok">Genereaza token</button></div><div id="tokMsg"></div>
  <table><thead><tr><th>Nume</th><th>Scope</th><th>Stare</th><th>Creat</th><th>Folosit</th><th></th></tr></thead><tbody>${tokens.map(t => `<tr><td>${esc(t.name || t.id)}</td><td>${esc(t.scope)}</td><td>${t.revoked ? '<span class="tag failed">revocat</span>' : '<span class="tag ok">activ</span>'}</td><td class="small">${fmtDate(t.created_at)}</td><td class="small">${fmtDate(t.last_used_at)}</td><td>${t.revoked ? '' : `<button class="btn small danger" data-revoke="${esc(t.id)}">Revoca</button>`}</td></tr>`).join('')}</tbody></table></div>
  <div class="card"><h2>Joburi de ingestie</h2><table><thead><tr><th>Job</th><th>Tip</th><th>Stare</th><th>Mesaj</th><th>Etape (ms)</th><th>Inceput</th></tr></thead><tbody>${jobs.map(j => `<tr><td class="code small">${esc(j.id)}<br>${esc(j.target_id)}</td><td>${esc(j.kind)}</td><td><span class="tag ${esc(j.status)}">${esc(j.status)}</span></td><td class="small">${esc(j.message)}</td><td class="small code">${esc(JSON.stringify(j.stages))}</td><td class="small">${fmtDate(j.started_at)}</td></tr>`).join('') || '<tr><td colspan="6" class="muted">Niciun job.</td></tr>'}</tbody></table></div>`;
  $('mkTok').onclick = async () => { try { const t = await api('/api/v1/admin/tokens', { json: { name: $('tokName').value, scope: $('tokScope').value } }); $('tokMsg').innerHTML = `<div class="notice good">Token nou (copiaza-l acum, nu va mai fi afisat):<br><span class="code">${esc(t.token)}</span></div>`; } catch (e) { toast(e.message); } };
  document.querySelectorAll('[data-revoke]').forEach(b => b.onclick = async () => { if (!confirm('Revoci tokenul?')) return; await api(`/api/v1/admin/tokens/${b.dataset.revoke}`, { method: 'DELETE' }); renderStatus(); });
}

boot();

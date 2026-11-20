// Exercise the actual settings-page JavaScript with an offline DOM/fetch stand-in.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../settings_page.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function element() {
  return { value: '', checked: false, children: [], append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; } };
}

async function setup({ legacy = false, modern = false } = {}) {
  const elements = new Map();
  const get = (id) => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); };
  const calls = [];
  let smalltalk = { ai: true, ratio: .2, daily_cap: 30 };
  const names = { kwonjun: 'kwonjun', itoa: 'ITOAR' };
  const states = Object.fromEntries(['kwonjun', 'itoa'].map(npc => [npc, {
    npc_name: names[npc], persona: `- 이름: ${names[npc]}`, persona_file: `${npc}.md`, ai: 'fake', rules: 'rules',
    system_preview: npc, speaker: 'Alice', speaker_known: true, players: {}, notes: [], history: [],
    settings: { auto_update: true, keep_history: false, brain_model: 'claude-haiku-5-5', stt_hints: '' },
    model_options: [
      {id: 'claude-haiku-5-5', name: 'Claude Haiku 5.5', cold_usd: .000575, warm_usd: .00023},
      {id: 'claude-sonnet-5-5', name: 'Claude Sonnet 5.5', cold_usd: .0115, warm_usd: .0043},
      {id: 'claude-opus-5-5', name: 'Claude Opus 5.5', cold_usd: .023, warm_usd: .0086},
    ],
    limits: { persona: 6000, note: 80, notes: 20, history: 10, impression: 150 },
  }]));
  if (legacy) {
    for (const state of Object.values(states)) delete state.model_options;
  }
  if (modern) {   // what the current app sends: effort, recall and memory compaction
    for (const state of Object.values(states)) {
      Object.assign(state.settings, { brain_effort: 'low', auto_recall: true, compact_on: true, compact_user_notes: false,
        compact_note_count: 50, compact_note_chars: 3000, compact_impression_chars: 400, compact_minutes: 10 });
      state.effort_options = ['low', 'medium', 'high', 'xhigh', 'max'].map(id => ({ id, name: id }));
      state.compaction = { running: false, backups: ['2026-11-19 20:10'], now: { count: 21, chars: 1480 },
        ranges: { compact_note_count: [5, 59], compact_note_chars: [200, 4800], compact_impression_chars: [100, 499], compact_minutes: [1, 1440] },
        last: { at: '2026-11-19 20:10', ok: true, error: '', notes_before: { count: 52, chars: 3120 }, notes_after: { count: 21, chars: 1480 },
                impressions: [{ name: 'Alice', before: 450, after: 190 }], usage: { in: 4000, out: 2000 }, cost_usd: 0.0014 } };
    }
  }
  let delay = null, fail = false;
  const context = vm.createContext({
    document: { getElementById: get, createElement: element }, confirm: () => true,
    setTimeout: () => 1, clearTimeout: () => {},
    fetch: async (url, options) => {
      const body = options.body ? JSON.parse(options.body) : null;
      const npc = body?.npc || new URL(url, 'http://local').searchParams.get('npc');
      calls.push({ url, options, body, npc });
      if (delay) await delay;
      if (fail) return { status: 400, ok: false, json: async () => ({error: 'invalid settings'}) };
      if (body) {
        const { settings, smalltalk: smalltalkPatch, ...rest } = body;
        Object.assign(states[npc], rest);
        if (settings) Object.assign(states[npc].settings, settings);
        if (smalltalkPatch) smalltalk = { ...smalltalk, ...smalltalkPatch };
      }
      return { status: 200, ok: true, json: async () => structuredClone({ ...states[npc], ...(legacy ? {} : { smalltalk }) }) };
    },
  });
  vm.runInContext(script, context);
  await new Promise(setImmediate);
  return { calls, get, context, delay: (promise) => { delay = promise; }, fail: value => { fail = value; } };
}

test('defaults to kwonjun with the token header', async () => {
  const t = await setup();
  assert.equal(t.calls[0].url, '/api/state?npc=kwonjun');
  assert.equal(t.calls[0].options.headers['X-W04-Token'], '__TOKEN__');
  assert.equal(t.get('npcName').value, 'kwonjun');
});

test('an already running older server keeps the new controls disabled', async () => {
  const t = await setup({legacy: true});
  assert.equal(t.get('npcName').value, 'kwonjun');
  assert.equal(t.get('voiceVersion').hidden, false);
  for (const id of ['brainModel', 'sttHints', 'saveModel', 'saveHints', 'saveSmalltalk']) assert.equal(t.get(id).disabled, true);
});

test('switching NPC reloads its persona and clears prior working edits', async () => {
  const t = await setup();
  vm.runInContext('players.Alice = {note:"kwon only"}; dirty.players = true; notes.push({text:"old"}); dirty.notes = true;', t.context);
  t.get('npcSelect').value = 'itoa';
  await t.get('npcSelect').onchange();
  assert.equal(t.calls.at(-1).url, '/api/state?npc=itoa');
  assert.equal(t.get('npcName').value, 'ITOAR');
  assert.equal(vm.runInContext('Object.keys(players).length + notes.length', t.context), 0);
});

test('all save payloads carry the chosen NPC', async () => {
  const t = await setup();
  t.get('npcSelect').value = 'itoa';
  await t.get('npcSelect').onchange();
  for (const body of [{ persona: '- 이름: 토아' }, { notes: [{ text: '귤' }] }, { players: {} },
                      { settings: { keep_history: true } }, { history_clear: true }]) {
    await vm.runInContext(`save(${JSON.stringify(body)})`, t.context);
    assert.equal(t.calls.at(-1).body.npc, 'itoa');
  }
});

test('blocks saves during an NPC reload', async () => {
  const t = await setup();
  let release;
  t.delay(new Promise(resolve => { release = resolve; }));
  t.get('npcSelect').value = 'itoa';
  const switching = t.get('npcSelect').onchange();
  const before = t.calls.length;
  assert.equal(await vm.runInContext('save({notes:[]})', t.context), false);
  assert.equal(t.calls.length, before);
  release();
  await switching;
});

test('discards a response from a save made before the NPC changed', async () => {
  const t = await setup();
  let release;
  t.delay(new Promise(resolve => { release = resolve; }));
  const saving = vm.runInContext('save({notes:[{text:"kwon memo"}]})', t.context);
  t.get('npcSelect').value = 'itoa';
  const switching = t.get('npcSelect').onchange();
  release();
  assert.equal(await saving, false);
  await switching;
  assert.equal(t.get('npcName').value, 'ITOAR');
});

test('shows three model estimates and saves model and nickname hints for only the selected NPC', async () => {
  const t = await setup();
  assert.equal(t.get('brainModel').children.length, 3);
  assert.equal(t.get('modelCosts').children.length, 3);
  assert.equal(t.get('modelCosts').children[0].children[1].children[0], '$0.000575');
  t.get('npcSelect').value = 'itoa';
  await t.get('npcSelect').onchange();
  t.get('brainModel').value = 'claude-opus-5-5';
  t.get('brainModel').onchange();
  await t.get('saveModel').onclick();
  assert.deepEqual(t.calls.at(-1).body, {npc: 'itoa', settings: {brain_model: 'claude-opus-5-5'}});
  t.get('sttHints').value = '404 -> 사공사\n이토아';
  t.get('sttHints').oninput();
  await t.get('saveHints').onclick();
  assert.deepEqual(t.calls.at(-1).body, {npc: 'itoa', settings: {stt_hints: '404 -> 사공사\n이토아'}});
  assert.equal(t.get('brainModel').value, 'claude-opus-5-5');
  t.get('npcSelect').value = 'kwonjun';
  await t.get('npcSelect').onchange();
  assert.equal(t.get('brainModel').value, 'claude-haiku-5-5');
  assert.equal(t.get('sttHints').value, '');
  t.get('npcSelect').value = 'itoa';
  await t.get('npcSelect').onchange();
  assert.equal(t.get('brainModel').value, 'claude-opus-5-5');
  assert.equal(t.get('sttHints').value, '404 -> 사공사\n이토아');
});

test('preserves unsaved hints and model through other saves, refreshes and failed saves', async () => {
  const t = await setup();
  t.get('sttHints').value = '404 -> 사공사';
  t.get('sttHints').oninput();
  t.get('brainModel').value = 'claude-sonnet-5-5';
  t.get('brainModel').onchange();
  await vm.runInContext('save({settings:{auto_update:false}})', t.context);
  await t.get('refresh').onclick();
  assert.equal(t.get('sttHints').value, '404 -> 사공사');
  assert.equal(t.get('brainModel').value, 'claude-sonnet-5-5');
  t.fail(true);
  await t.get('saveHints').onclick();
  await t.get('saveModel').onclick();
  assert.equal(vm.runInContext('dirty.hints && dirty.model', t.context), true);
  assert.equal(t.get('sttHints').value, '404 -> 사공사');
  assert.equal(t.get('brainModel').value, 'claude-sonnet-5-5');
  t.fail(false);
  t.get('npcSelect').value = 'itoa';
  await t.get('npcSelect').onchange();
  assert.equal(t.get('sttHints').value, '');
  assert.equal(t.get('brainModel').value, 'claude-haiku-5-5');
});

test('saving empty hints clears them and late saves cannot repaint another NPC', async () => {
  const t = await setup();
  t.get('sttHints').value = '404 -> 사공사';
  t.get('sttHints').oninput();
  await t.get('saveHints').onclick();
  t.get('sttHints').value = '';
  t.get('sttHints').oninput();
  await t.get('saveHints').onclick();
  assert.equal(t.calls.at(-1).body.settings.stt_hints, '');
  let release;
  t.delay(new Promise(resolve => { release = resolve; }));
  t.get('brainModel').value = 'claude-opus-5-5';
  t.get('brainModel').onchange();
  const saving = t.get('saveModel').onclick();
  t.get('npcSelect').value = 'itoa';
  const switching = t.get('npcSelect').onchange();
  release();
  await saving;
  await switching;
  assert.equal(t.get('brainModel').value, 'claude-haiku-5-5');
});

test('both NPCs show the same small-talk settings and reject invalid input', async () => {
  const t = await setup();
  assert.equal(t.get('smalltalkAI').checked, true);
  assert.equal(t.get('smalltalkRatio').value, '20');
  assert.equal(t.get('smalltalkCap').value, '30');
  t.get('smalltalkRatio').value = '101';
  await t.get('saveSmalltalk').onclick();
  assert.equal(t.calls.length, 1);
  t.get('smalltalkAI').checked = false;
  t.get('smalltalkRatio').value = '12.5';
  t.get('smalltalkCap').value = '0';
  t.get('smalltalkRatio').oninput();
  await t.get('refresh').onclick();
  assert.equal(t.get('smalltalkRatio').value, '12.5');
  await t.get('saveSmalltalk').onclick();
  assert.deepEqual(t.calls.at(-1).body, { npc: 'kwonjun', smalltalk: { ai: false, ratio: .125, daily_cap: 0 } });
  t.get('npcSelect').value = 'itoa';
  await t.get('npcSelect').onchange();
  assert.equal(t.get('smalltalkAI').checked, false);
  assert.equal(t.get('smalltalkRatio').value, '12.5');
  assert.equal(t.get('smalltalkCap').value, '0');
});

test('an older running app keeps effort, recall and compaction controls disabled', async () => {
  const t = await setup();
  for (const id of ['brainEffort', 'saveEffort', 'autoRecall', 'compactOn', 'saveCompact', 'compactRun', 'compactUndo'])
    assert.equal(t.get(id).disabled, true);
  assert.equal(t.get('compactVersion').hidden, false);
});

test('effort and recall are saved for the selected NPC', async () => {
  const t = await setup({ modern: true });
  assert.equal(t.get('brainEffort').children.length, 5);
  assert.equal(t.get('brainEffort').value, 'low');
  assert.equal(t.get('autoRecall').checked, true);
  t.get('brainEffort').value = 'high';
  t.get('brainEffort').onchange();
  await t.get('refresh').onclick();
  assert.equal(t.get('brainEffort').value, 'high');          // an unsaved choice survives a refresh
  await t.get('saveEffort').onclick();
  assert.deepEqual(t.calls.at(-1).body, { npc: 'kwonjun', settings: { brain_effort: 'high' } });
  t.get('autoRecall').checked = false;
  await t.get('autoRecall').onchange();
  assert.deepEqual(t.calls.at(-1).body, { npc: 'kwonjun', settings: { auto_recall: false } });
});

test('compaction shows last time and sizes, validates thresholds, runs and undoes', async () => {
  const t = await setup({ modern: true });
  const status = t.get('compactStatus').textContent;
  for (const piece of ['지금: 메모 21개 1,480자', '마지막 정리: 2026-11-19 20:10', '메모 52개 3,120자 → 메모 21개 1,480자',
                       'Alice의 인상 450자 → 190자', '$0.0014', '되돌릴 수 있는 정리 기록 1개'])
    assert.ok(status.includes(piece), piece + ' / ' + status);
  assert.equal(t.get('compactCount').value, '50');
  assert.equal(t.get('compactChars').value, '3000');
  assert.equal(t.get('compactImpression').value, '400');
  assert.equal(t.get('compactMinutes').value, '10');
  assert.equal(t.get('compactUndo').disabled, false);
  const before = t.calls.length;
  t.get('compactCount').value = '60';
  t.get('compactCount').oninput();
  await t.get('saveCompact').onclick();
  assert.equal(t.calls.length, before);                       // out of range: nothing is sent
  t.get('compactCount').value = '40';
  t.get('compactUser').checked = true;
  await t.get('saveCompact').onclick();
  assert.deepEqual(t.calls.at(-1).body, { npc: 'kwonjun', settings: { compact_on: true, compact_user_notes: true,
    compact_note_count: 40, compact_note_chars: 3000, compact_impression_chars: 400, compact_minutes: 10 } });
  await t.get('compactRun').onclick();
  assert.deepEqual(t.calls.at(-1).body, { npc: 'kwonjun', compact: 'run' });
  await t.get('compactUndo').onclick();
  assert.deepEqual(t.calls.at(-1).body, { npc: 'kwonjun', compact: 'undo' });
});

test('memo and impression saves send what the page started from and keep the draft when the save fails', async () => {
  const t = await setup();
  vm.runInContext('S.notes = [{text:"하나", by:"user", at:""}]; S.players = {Alice: {note:"다정함", locked:false, talks:1, updated:""}}; render(true);', t.context);
  vm.runInContext('notes[0].text = "하나 고침"; dirty.notes = true; players.Alice.note = "웃김"; dirty.players = true;', t.context);
  t.fail(true);
  await t.get('saveNotes').onclick();
  assert.equal(vm.runInContext('dirty.notes', t.context), true);
  assert.equal(vm.runInContext('notes[0].text', t.context), '하나 고침');
  t.fail(false);
  await t.get('saveNotes').onclick();
  assert.deepEqual(t.calls.at(-1).body.notes_base, ['하나']);
  assert.equal(t.calls.at(-1).body.notes[0].text, '하나 고침');
  assert.equal(vm.runInContext('dirty.notes', t.context), false);
  assert.deepEqual(vm.runInContext('baseNotes', t.context), ['하나 고침']);
  await t.get('savePlayers').onclick();
  assert.equal(t.calls.at(-1).body.players_base.Alice.note, '다정함');
  assert.equal(t.calls.at(-1).body.players.Alice.note, '웃김');
  assert.equal(vm.runInContext('dirty.players', t.context), false);
});

test('switching NPC asks before dropping unsaved edits and cannot save a draft under the other NPC', async () => {
  const t = await setup();
  vm.runInContext('notes.push({text:"kwon draft", by:"user", at:""}); dirty.notes = true;', t.context);
  const asked = [];
  t.context.confirm = (msg) => { asked.push(msg); return false; };
  t.get('npcSelect').value = 'itoa';
  const before = t.calls.length;
  await t.get('npcSelect').onchange();
  assert.equal(asked.length, 1);
  assert.equal(t.calls.length, before);
  assert.equal(t.get('npcSelect').value, 'kwonjun');
  assert.equal(vm.runInContext('notes.length', t.context), 1);
  // a draft typed while the other NPC loads is replaced together with the state, never kept for the new NPC
  t.context.confirm = () => true;
  let release;
  t.delay(new Promise(resolve => { release = resolve; }));
  t.get('npcSelect').value = 'itoa';
  const switching = t.get('npcSelect').onchange();
  assert.equal(t.get('main').inert, true);
  vm.runInContext('notes.push({text:"typed late"}); dirty.notes = true;', t.context);
  release();
  await switching;
  assert.equal(t.get('main').inert, false);
  assert.equal(vm.runInContext('notes.length', t.context), 0);
  assert.equal(vm.runInContext('owner.notes', t.context), 'itoa');
  // a working copy that belongs to another NPC is refused
  vm.runInContext('owner.notes = "kwonjun"; notes.push({text:"wrong owner"}); dirty.notes = true;', t.context);
  const count = t.calls.length;
  assert.equal(await t.get('saveNotes').onclick(), false);
  assert.equal(t.calls.length, count);
});

test('forgetting one talk asks first; counters show the limits', async () => {
  const t = await setup();
  vm.runInContext('S.history = [{heard:"안녕", line:"응"}]; S.notes = [{text:"귤", by:"ai", at:""}]; render(false);', t.context);
  assert.equal(t.get('notesCount').textContent, '메모 1 / 20개 · 한 줄 80자까지');
  const del = t.get('history').children[0].children[1];
  let answer = false;
  t.context.confirm = () => answer;
  const before = t.calls.length;
  del.onclick();
  await new Promise(setImmediate);
  assert.equal(t.calls.length, before);
  answer = true;
  del.onclick();
  await new Promise(setImmediate);
  assert.equal(t.calls.at(-1).body.history_delete, 0);
});

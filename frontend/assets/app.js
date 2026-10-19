const form = document.querySelector('#question-form');
const questionInput = document.querySelector('#question');
const characterCount = document.querySelector('#character-count');
const submitButton = document.querySelector('#submit-button');
const statusPill = document.querySelector('#service-status');
const statusText = document.querySelector('#service-status-text');
const response = document.querySelector('#response');
const responseLabel = document.querySelector('#response-label');
const placeholder = document.querySelector('#response-placeholder');
const responseMessage = document.querySelector('#response-message');
const answerText = document.querySelector('#answer-text');
const sources = document.querySelector('#sources');
const sourceList = document.querySelector('#source-list');
const disclaimer = document.querySelector('#answer-disclaimer');

let assistantReady = false;
let requestPending = false;

function setServiceStatus(status, message) {
  statusPill.className = `service-status ${status}`;
  statusText.textContent = message;
  assistantReady = status === 'ready' || status === 'caution';
  submitButton.disabled = !assistantReady || requestPending;
}

async function refreshServiceStatus() {
  try {
    const result = await fetch('/health', { cache: 'no-store' });
    if (!result.ok) throw new Error('The service status could not be loaded.');
    const health = await result.json();

    if (health.status === 'loading') {
      setServiceStatus('waiting', 'Loading local models');
      return;
    }
    if (health.status === 'error') {
      setServiceStatus('failed', 'Setup needs attention');
      return;
    }
    if (health.status === 'degraded') {
      setServiceStatus('caution', 'Ready · not calibrated');
      return;
    }
    setServiceStatus('ready', 'Assistant ready');
  } catch {
    setServiceStatus('failed', 'Service unavailable');
  }
}

function showMessage(message, kind = 'info') {
  placeholder.hidden = true;
  responseMessage.hidden = false;
  responseMessage.className = `response-message ${kind}`;
  responseMessage.textContent = message;
  answerText.hidden = true;
  sources.hidden = true;
  disclaimer.hidden = true;
  responseLabel.textContent = kind === 'error' ? 'PLEASE TRY AGAIN' : 'ASSISTANT STATUS';
}

function responseKind(result) {
  if (result.emergency) return 'SAFETY CHECK · EMERGENCY';
  if (result.redirected) return 'SAFETY CHECK · PERSONAL ADVICE';
  if (result.refused) return 'SAFETY CHECK · REFUSAL';
  if (result.abstained) {
    if (result.abstain_reason === 'uncalibrated') return 'ABSTAINED · NOT CALIBRATED';
    if (result.abstain_reason === 'citation_validation_failed') return 'ANSWER WITHHELD · CITATION CHECK';
    return 'OUT OF SCOPE OR NOT IN SOURCES';
  }
  return 'ANSWER WITH SOURCES';
}

function safeSourceLink(url) {
  try {
    const parsed = new URL(url);
    return ['https:', 'http:'].includes(parsed.protocol) ? parsed.href : null;
  } catch {
    return null;
  }
}

function sourceSearchLink(source) {
  const publisher = String(source.source || '');
  const officialSites = publisher.includes('NIDDK')
    ? 'site:niddk.nih.gov'
    : publisher.includes('NIHSeniorHealth')
      ? '(site:medlineplus.gov OR site:nia.nih.gov)'
      : '';
  const query = [officialSites, source.title].filter(Boolean).join(' ');
  return `https://www.google.com/search?q=${encodeURIComponent(query)}`;
}

function appendFormattedText(parent, text) {
  const boldParts = text.split(/(\*\*[^*]+\*\*)/g);
  for (const part of boldParts) {
    if (part.startsWith('**') && part.endsWith('**')) {
      const strong = document.createElement('strong');
      strong.textContent = part.slice(2, -2);
      parent.append(strong);
    } else {
      parent.append(document.createTextNode(part));
    }
  }
}

function renderFormattedAnswer(text) {
  const fragment = document.createDocumentFragment();
  const lines = String(text).split(/\r?\n/);
  let paragraphLines = [];
  let list = null;

  function flushParagraph() {
    if (!paragraphLines.length) return;
    const paragraph = document.createElement('p');
    paragraphLines.forEach((line, index) => {
      appendFormattedText(paragraph, line);
      if (index < paragraphLines.length - 1) paragraph.append(document.createElement('br'));
    });
    fragment.append(paragraph);
    paragraphLines = [];
  }

  for (const line of lines) {
    const bullet = line.match(/^\s*[-*]\s+(.+?)\s*$/);
    if (bullet) {
      flushParagraph();
      if (!list) {
        list = document.createElement('ul');
        fragment.append(list);
      }
      const item = document.createElement('li');
      appendFormattedText(item, bullet[1]);
      list.append(item);
    } else if (!line.trim()) {
      flushParagraph();
      list = null;
    } else {
      list = null;
      paragraphLines.push(line);
    }
  }
  flushParagraph();
  return fragment;
}

function renderSources(items) {
  sourceList.replaceChildren();
  if (!items?.length) {
    sources.hidden = true;
    return;
  }

  for (const source of items) {
    const item = document.createElement('li');
    const title = document.createElement('span');
    title.className = 'source-title';
    title.textContent = `[${source.n}] ${source.title || 'Untitled source'}`;
    const href = safeSourceLink(source.url);
    if (href) {
      const link = document.createElement('a');
      link.className = title.className;
      link.href = href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = title.textContent;
      item.append(link);
    } else {
      item.append(title);
    }
    const meta = document.createElement('span');
    meta.className = 'source-meta';
    meta.textContent = source.source || 'Health information source';
    item.append(meta);

    const actions = document.createElement('div');
    actions.className = 'source-actions';
    if (String(source.source || '').startsWith('MedQuAD')) {
      const search = document.createElement('a');
      search.href = sourceSearchLink(source);
      search.target = '_blank';
      search.rel = 'noopener noreferrer';
      search.textContent = 'Search for a current page';
      actions.append(search);
    }
    if (actions.childElementCount) item.append(actions);

    if (source.excerpt) {
      const details = document.createElement('details');
      details.className = 'source-excerpt';
      const summary = document.createElement('summary');
      summary.textContent = 'View retrieved excerpt';
      const excerpt = document.createElement('p');
      excerpt.textContent = source.excerpt;
      details.append(summary, excerpt);
      item.append(details);
    }
    sourceList.append(item);
  }
  sources.hidden = false;
}

function renderAnswer(result) {
  placeholder.hidden = true;
  responseMessage.hidden = true;
  responseLabel.textContent = responseKind(result);
  answerText.replaceChildren(renderFormattedAnswer(result.answer || 'No answer was returned.'));
  answerText.hidden = false;
  renderSources(result.sources);
  disclaimer.textContent = result.disclaimer || '';
  disclaimer.hidden = !result.disclaimer;
}

questionInput.addEventListener('input', () => {
  characterCount.textContent = `${questionInput.value.length.toLocaleString()} / 1,000`;
});

document.querySelectorAll('[data-question]').forEach((button) => {
  button.addEventListener('click', () => {
    questionInput.value = button.dataset.question;
    questionInput.dispatchEvent(new Event('input'));
    questionInput.focus();
  });
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || !assistantReady || requestPending) return;

  requestPending = true;
  submitButton.disabled = true;
  submitButton.querySelector('span:first-child').textContent = 'Working…';
  response.setAttribute('aria-busy', 'true');
  responseLabel.textContent = 'CHECKING THE SOURCES';
  placeholder.hidden = true;
  responseMessage.hidden = true;
  answerText.hidden = true;
  sources.hidden = true;
  disclaimer.hidden = true;

  try {
    const result = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await result.json().catch(() => ({}));
    if (!result.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : 'The request could not be completed.';
      throw new Error(detail);
    }
    renderAnswer(data);
  } catch (error) {
    showMessage(error.message || 'The request could not be completed. Please try again.', 'error');
    if (error.message?.toLowerCase().includes('still loading')) {
      await refreshServiceStatus();
    }
  } finally {
    requestPending = false;
    response.setAttribute('aria-busy', 'false');
    submitButton.querySelector('span:first-child').textContent = 'Ask the assistant';
    submitButton.disabled = !assistantReady;
  }
});

refreshServiceStatus();
const statusRefresh = window.setInterval(async () => {
  await refreshServiceStatus();
  if (assistantReady || statusPill.classList.contains('failed')) {
    window.clearInterval(statusRefresh);
  }
}, 4000);

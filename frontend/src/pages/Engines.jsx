import './Engines.css';

const ENGINE_DATA = [
  {
    id: 'ml',
    name: '1. The AI Brain (Machine Learning)',
    weight: '60% of Score',
    meaning: "This is our main brain. We showed it thousands of news articles that human experts already labeled as 'Biased' or 'Neutral'. By reading them, it learned exactly which combination of words are used to push a political agenda.",
    example: 'The heroic whistleblower exposed the tyrannical administration.',
    highlight: ['heroic', 'tyrannical'],
    explanation: "The AI flags this because real journalists say 'The employee leaked documents', not 'heroic whistleblower'."
  },
  {
    id: 'emotion',
    name: '2. Emotional Intensity',
    weight: '10% of Score',
    meaning: "Good news should be boring and factual. This engine uses a massive dictionary of words to measure if an article is trying to make you feel angry, scared, or excited. It mathematically adds up the emotional 'charge' of every word.",
    example: 'The completely devastating and catastrophic failure of the policy...',
    highlight: ['devastating', 'catastrophic'],
    explanation: "Words like 'devastating' and 'catastrophic' are highly emotional and trigger a penalty for sensationalism."
  },
  {
    id: 'subjectivity',
    name: '3. Fact vs Opinion (Subjectivity)',
    weight: '10% of Score',
    meaning: "How do we know if something is a fact or an opinion? This engine scans the text for adjectives and personal judgments. It scores the article based on the ratio of hard facts (like numbers and dates) to personal opinions.",
    example: 'Factual: The bill passed 51-49.\nOpinion: The bill was a terrible mistake.',
    highlight: ['terrible', 'mistake'],
    explanation: "The first sentence is a verifiable fact. The second sentence uses opinionated adjectives, making it highly subjective."
  },
  {
    id: 'passive',
    name: '4. Dodging Blame (Passive Voice)',
    weight: '10% of Score',
    meaning: "Politicians and biased journalists use 'passive voice' to hide who is responsible for a mistake. This engine scans the grammar of the sentence to catch when someone is dodging the blame.",
    example: 'Active: The CEO lost the funds.\nPassive: Mistakes were made and funds were lost.',
    highlight: ['Mistakes were made', 'funds were lost'],
    explanation: "The passive sentence completely hides WHO made the mistakes. We penalize this."
  },
  {
    id: 'loaded',
    name: '5. Loaded Language',
    weight: '5% of Score',
    meaning: "This engine uses a strict list of 'red flag' words. These are words that are almost never used in objective reporting because they are specifically designed to manipulate the reader's opinion.",
    example: 'The radical extremists pushed their woke propaganda.',
    highlight: ['radical', 'extremists', 'woke', 'propaganda'],
    explanation: "These words are instantly flagged as manipulative because they force an opinion onto the reader."
  },
  {
    id: 'hedge',
    name: '6. Rumors & Fake Certainty (Hedges)',
    weight: '5% of Score',
    meaning: "This catches two tricks: 1) Pretending a rumor is a fact (fake certainty), or 2) Making a wild claim without proof (evading accountability).",
    example: 'The candidate reportedly has absolutely no chance of winning.',
    highlight: ['reportedly', 'absolutely'],
    explanation: "'Absolutely' fakes certainty, while 'reportedly' is a cowardly way to spread a rumor without naming a real source."
  }
];

export default function Engines() {
  const renderExample = (text, highlights) => {
    const escapeRegExp = (string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    
    return text.split('\n').map((line, i) => {
      let htmlLine = line;
      highlights.forEach(word => {
        const regex = new RegExp(`(${escapeRegExp(word)})`, 'gi');
        htmlLine = htmlLine.replace(regex, '<span class="highlight-word">$1</span>');
      });
      return (
        <div key={i} dangerouslySetInnerHTML={{ __html: htmlLine }} style={{ marginBottom: line.trim() ? '0.25rem' : '0' }} />
      );
    });
  };

  return (
    <div className="page-container engines-page">
      <div className="engines-header card">
        <h1 className="engines-title">How It Works</h1>
        <p className="engines-subtitle">Understanding the 6-Engine NLP Pipeline</p>
        <p className="engines-desc">
          SLANT does not use a single "black box" AI. Instead, it breaks every article down using six specialized linguistic engines. 
          Here is exactly what each engine looks for when analyzing an article.
        </p>
      </div>

      <div className="engines-grid">
        {ENGINE_DATA.map(engine => (
          <div key={engine.id} className="card engine-card">
            <div className="ec-header">
              <h2 className="ec-title">{engine.name}</h2>
              <span className="ec-weight">{engine.weight}</span>
            </div>
            <p className="ec-meaning">{engine.meaning}</p>
            
            <div className="ec-example-box">
              <div className="ec-example-label">Example Detection</div>
              <div className="ec-example-text">
                {renderExample(engine.example, engine.highlight)}
              </div>
              <div className="ec-explanation">
                <strong>Why it's flagged: </strong>
                {engine.explanation}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

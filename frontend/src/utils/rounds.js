export function groupIntoRounds(xaiLogs, answeredFollowups = []) {
  const rounds = [];
  let currentRound = { roundNum: 1, logs: [], qa: [] };
  
  const specQA = answeredFollowups.filter(qa => qa.isSpecialist);

  xaiLogs.forEach(log => {
    // A new round starts when we hit 'debate' and we've already done an 'improved_diagnosis' or 'specialists_follow_up' in the current round
    if (log.stage === 'debate' && currentRound.logs.some(l => l.stage === 'improved_diagnosis' || l.stage === 'specialists_follow_up' || l.stage === 'choice')) {
      rounds.push(currentRound);
      currentRound = { roundNum: rounds.length + 1, logs: [], qa: [] };
    }
    currentRound.logs.push(log);
  });
  
  if (currentRound.logs.length > 0) {
    rounds.push(currentRound);
  }

  // Assign Q&A to the appropriate round.
  // Legacy chat histories might not have round tags on Q&A, so we associate Q&A with Round 1 by default,
  // or distribute them if we start adding `round` property to Q&A in the future.
  specQA.forEach(qa => {
    const roundIndex = qa.round ? Math.max(0, qa.round - 1) : 0;
    if (rounds[roundIndex]) {
      rounds[roundIndex].qa.push(qa);
    } else if (rounds.length > 0) {
      rounds[rounds.length - 1].qa.push(qa);
    }
  });

  return rounds;
}

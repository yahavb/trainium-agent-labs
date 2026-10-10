"""Small evidence ledger; changes in error class are progress hypotheses, not fixes."""
from failure_selection import classify_failure,code_fingerprint

class RepairHistory:
    def __init__(self):self.entries=[];self.best=None
    def observe(self,source,feedback,reward,verified_shapes=0):
        diagnostic=classify_failure(feedback)
        prior=self.entries[-1] if self.entries else None
        entry=dict(signature=diagnostic.normalized_signature,category=diagnostic.failure_category,hash=code_fingerprint(source),reward=reward,verified_shapes=verified_shapes,source_changed=(prior['hash']!=code_fingerprint(source)) if prior else None,previous_issue_no_longer_first=(prior['signature']!=diagnostic.normalized_signature) if prior else None)
        self.entries.append(entry);self.entries=self.entries[-6:]
        if self.best is None or (verified_shapes,reward)>(self.best['verified_shapes'],self.best['reward']):self.best=dict(entry,source=source,feedback=feedback)
        return entry
    def guidance(self):
        if len(self.entries)<2:return ''
        last=self.entries[-1];same=[e for e in self.entries if e['signature']==last['signature']]
        if len(same)<2:return ''
        count=len(same)
        scope='coordinated changes to the failing allocation, transfer and consumers' if count>=2 else 'localized correction'
        if count>=4 and last['category'] in ('NUMERICAL_MISMATCH','INCOMPLETE_OUTPUT'):scope='redesign of only the failing algorithm/dataflow'
        return f"Repair history: this normalized failure occurred {count} times; last source changed={last['source_changed']}. The previous correction did not remove the first reported failure. Use {scope}; preserve unrelated verified behavior. A changed error is not proof that the earlier error is fixed."

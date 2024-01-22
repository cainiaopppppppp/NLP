Trial Data for SemEval-2 Task #8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals
Iris Hendrickx, Su Nam Kim, Zornitsa Kozareva, Preslav Nakov, Diarmuid Ó Séaghdha, Sebastian Padó, Marco Pennacchiotti, Lorenza Romano and Stan Szpakowicz. The accompanying dataset is released under a Creative Commons Atrribution 3.0 Unported Licence (http://creativecommons.org/licenses/by/3.0/).
Version 1.0: 30/9/09


SUMMARY

This dataset consists of 934 sentences that have been annotated according to the scheme for SemEval-2 Task #8. The sentences were originally collected as part of the data for SemEval-1 Task #4 (Classification of Semantic Relations between Nominals) but have been reannotated with the new relations and relation definitions.


RELATIONS

The relations that have been chosen for SemEval-2 Task #8 are the following:

(1) Cause-Effect
(2) Instrument-Agency
(3) Product-Producer
(4) Content-Container
(5) Entity-Origin
(6) Entity-Destination
(7) Component-Whole
(8) Member-Collection
(9) Communication-Topic

Relations 1-5 were also used in SemEval-1 Task #4; the trial dataset has been compiled from the positive and negative examples for these five relations. All nine relations were used in annotating the dataset but due to its origin, relations 1-5 are the most frequent. The definitions for these relations are provided in the files rel*_def.htm. 


DATA FORMAT

The format of the data is illustrated by the following example:

102 "Diabetes drug shows promise for preventing <e1>brain injury</e1> from <e2>radiation therapy</e2>."
Cause-Effect(e2,e1)
Comment: brain injury is an event, radiation therapy leads to brain injury.

The first line contains the sentence itself, preceded by a numerical identifier (in this case, 102). Each sentence is annotated with three pieces of information:

(a) Two entity mentions in the sentence are marked up as e1 and e2 - the numbering simply reflects the order of the mentions.
(b) If one of the semantic relations 1-9 holds between e1 and e2, the sentence is labelled with this relation and the order in which the relation arguments are filled by e1 and e2. For example, Cause-Effect(e1,e2) means that e1 is the Cause and e2 is the Effect, whereas Cause-Effect(e2,e1) means that e2 is the Cause and e1 is the Effect. If none of the relations 1-9 hold, the sentence is labelled "Other". In total, there are 19 possible labels, though not all appear in the trial data.
(c) A comment may be provided to explain why the annotators chose a given label. Comments are intended for human readers and should be ignored by automatic systems participating in the task.


EVALUATION

The task is to predict, given a sentence and two marked-up entities, which of the relation labels to apply. Hence, the gold-standard labels (Cause-Effect(e1,e2) etc.) should be provided to a trial system at training time but not at test time. The official evaluation measures are accuracy over all examples and macro-averaged F-score over the 18 relation labels apart from Other. To calculate the F-score, 18 individual F-scores - one for each relation label - are calculated in the standard way and the average of these scores is taken. For each relation Rel, each sentence labelled Rel in the gold standard will count as either a true positive or a false negative, depending on whether it was correctly labelled by the system; each sentence labelled with a different relation or with Other will count as a true negative or false positive.

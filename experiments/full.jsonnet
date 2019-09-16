local caf = import "credences_and_features.jsonnet";
local bert = import "bert_only.jsonnet";

bert + {'model': caf['model'] + bert['model']}

/*
1039 -> 1:
1039 -> 30 -> 1:
1039 -> 100 -> 10 -> 1:
 */

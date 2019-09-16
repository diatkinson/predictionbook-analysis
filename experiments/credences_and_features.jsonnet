local credences = import "credences.jsonnet";

credences + {
  "model" +: {
    "use_features": true,
  },
}

/*
15->1 ff: 0.186
15->5->1 ff: 0.187
15->10->5-$ ff: 0.186
*/

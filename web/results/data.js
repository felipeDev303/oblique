window.RESULTS = {
 "c1_vs_c1s": {
  "run": "R-20261103-01",
  "conditions": {
   "C1": {
    "cu": {
     "mean": 0.1638,
     "lo": 0.1081,
     "hi": 0.2224,
     "n": 49
    },
    "q_controlled": {
     "mean": 0.2054,
     "lo": 0.134,
     "hi": 0.2925,
     "n": 50
    },
    "d": {
     "mean": 0.6412,
     "lo": 0.6179,
     "hi": 0.6633,
     "n": 49
    },
    "sets": 49
   },
   "C1s": {
    "cu": {
     "mean": 0.4041,
     "lo": 0.3309,
     "hi": 0.4782,
     "n": 49
    },
    "q_controlled": {
     "mean": 0.4679,
     "lo": 0.3824,
     "hi": 0.5506,
     "n": 50
    },
    "d": {
     "mean": 0.6412,
     "lo": 0.6219,
     "hi": 0.661,
     "n": 49
    },
    "sets": 49
   }
  },
  "delta": {
   "cu": {
    "mean": 0.2403,
    "lo": 0.17,
    "hi": 0.3115,
    "n": 49
   },
   "q_controlled": {
    "mean": 0.2624,
    "lo": 0.1584,
    "hi": 0.3522,
    "n": 50
   },
   "q_gold": {
    "mean": 0.051,
    "lo": 0.0101,
    "hi": 0.0986,
    "n": 49
   },
   "d": {
    "mean": -0.0,
    "lo": -0.0227,
    "hi": 0.0227,
    "n": 49
   },
   "d_embed": {
    "mean": -0.0352,
    "lo": -0.0572,
    "hi": -0.0132,
    "n": 49
   }
  },
  "n_boot": 2000
 },
 "map": {
  "run": "R-20261031-01",
  "delta": 0.05,
  "c2": {
   "q_controlled": {
    "mean": 0.5303,
    "lo": 0.4055,
    "hi": 0.6551,
    "n": 50
   },
   "d": {
    "mean": 0.18,
    "lo": 0.1417,
    "hi": 0.22,
    "n": 50
   }
  },
  "ml": [
   "OB-001",
   "OB-004",
   "OB-005",
   "OB-007",
   "OB-008",
   "OB-009",
   "OB-011",
   "OB-012",
   "OB-013",
   "OB-016",
   "OB-018",
   "OB-020",
   "OB-022",
   "OB-023",
   "OB-025",
   "OB-026",
   "OB-027",
   "OB-029"
  ],
  "cards": [
   {
    "id": "OB-001",
    "text": "Remove the assumption every document shares.",
    "origin": "seed",
    "family": "subtraction",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.1976,
     "lo": -0.3636,
     "hi": -0.032,
     "n": 50
    },
    "dd": {
     "mean": 0.2177,
     "lo": 0.1582,
     "hi": 0.2755,
     "n": 49
    }
   },
   {
    "id": "OB-002",
    "text": "Solve it without the ingredient every method relies on.",
    "origin": "seed",
    "family": "subtraction",
    "band": "destructive",
    "confirmed": false,
    "dq": {
     "mean": -0.3416,
     "lo": -0.4906,
     "hi": -0.1868,
     "n": 50
    },
    "dd": {
     "mean": 0.2667,
     "lo": 0.2133,
     "hi": 0.3184,
     "n": 50
    }
   },
   {
    "id": "OB-003",
    "text": "Keep only the claim that would survive if the rest were false.",
    "origin": "seed",
    "family": "subtraction",
    "band": "inert",
    "confirmed": false,
    "dq": {
     "mean": -0.1128,
     "lo": -0.2715,
     "hi": 0.0492,
     "n": 50
    },
    "dd": {
     "mean": 0.0267,
     "lo": -0.0183,
     "hi": 0.0733,
     "n": 50
    }
   },
   {
    "id": "OB-004",
    "text": "Delete the benchmark; what question is left?",
    "origin": "seed",
    "family": "subtraction",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.0883,
     "lo": -0.2681,
     "hi": 0.088,
     "n": 50
    },
    "dd": {
     "mean": 0.0933,
     "lo": 0.0417,
     "hi": 0.1433,
     "n": 50
    }
   },
   {
    "id": "OB-005",
    "text": "Treat the reported outcome as the cause, not the effect.",
    "origin": "seed",
    "family": "inversion",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.1884,
     "lo": -0.3651,
     "hi": -0.0091,
     "n": 50
    },
    "dd": {
     "mean": 0.1327,
     "lo": 0.0782,
     "hi": 0.1871,
     "n": 49
    }
   },
   {
    "id": "OB-006",
    "text": "Argue for the conclusion the documents reject.",
    "origin": "seed",
    "family": "inversion",
    "band": "destructive",
    "confirmed": false,
    "dq": {
     "mean": -0.2648,
     "lo": -0.4287,
     "hi": -0.096,
     "n": 50
    },
    "dd": {
     "mean": 0.33,
     "lo": 0.2816,
     "hi": 0.3783,
     "n": 50
    }
   },
   {
    "id": "OB-007",
    "text": "Swap what is measured with what is held fixed.",
    "origin": "seed",
    "family": "inversion",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.1165,
     "lo": -0.2843,
     "hi": 0.0559,
     "n": 50
    },
    "dd": {
     "mean": 0.1667,
     "lo": 0.1059,
     "hi": 0.2274,
     "n": 48
    }
   },
   {
    "id": "OB-008",
    "text": "Read the documents as evidence against their own claims.",
    "origin": "seed",
    "family": "inversion",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": 0.0088,
     "lo": -0.1588,
     "hi": 0.1758,
     "n": 50
    },
    "dd": {
     "mean": 0.2,
     "lo": 0.1467,
     "hi": 0.2617,
     "n": 50
    }
   },
   {
    "id": "OB-009",
    "text": "Shrink the question to a single observation.",
    "origin": "seed",
    "family": "scale",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.0615,
     "lo": -0.2474,
     "hi": 0.1378,
     "n": 50
    },
    "dd": {
     "mean": 0.1083,
     "lo": 0.0617,
     "hi": 0.1533,
     "n": 50
    }
   },
   {
    "id": "OB-010",
    "text": "Ask what this means for the whole field at once.",
    "origin": "seed",
    "family": "scale",
    "band": "destructive",
    "confirmed": false,
    "dq": {
     "mean": -0.1868,
     "lo": -0.3174,
     "hi": -0.0577,
     "n": 50
    },
    "dd": {
     "mean": 0.0417,
     "lo": -0.0033,
     "hi": 0.0884,
     "n": 50
    }
   },
   {
    "id": "OB-011",
    "text": "Study the tail, not the average.",
    "origin": "seed",
    "family": "scale",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.1314,
     "lo": -0.3195,
     "hi": 0.0621,
     "n": 50
    },
    "dd": {
     "mean": 0.2622,
     "lo": 0.2014,
     "hi": 0.3212,
     "n": 48
    }
   },
   {
    "id": "OB-012",
    "text": "Change the unit of analysis by one level.",
    "origin": "seed",
    "family": "scale",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": 0.0335,
     "lo": -0.1518,
     "hi": 0.2163,
     "n": 50
    },
    "dd": {
     "mean": 0.1333,
     "lo": 0.0733,
     "hi": 0.1933,
     "n": 50
    }
   },
   {
    "id": "OB-013",
    "text": "The method becomes the object of study.",
    "origin": "seed",
    "family": "role",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": 0.0064,
     "lo": -0.1779,
     "hi": 0.1951,
     "n": 50
    },
    "dd": {
     "mean": 0.1933,
     "lo": 0.1283,
     "hi": 0.2633,
     "n": 50
    }
   },
   {
    "id": "OB-014",
    "text": "Give the dataset the job the model was doing.",
    "origin": "seed",
    "family": "role",
    "band": "destructive",
    "confirmed": false,
    "dq": {
     "mean": -0.2628,
     "lo": -0.4118,
     "hi": -0.1145,
     "n": 50
    },
    "dd": {
     "mean": 0.1888,
     "lo": 0.1378,
     "hi": 0.2398,
     "n": 49
    }
   },
   {
    "id": "OB-015",
    "text": "Write the idea for the people who produce the data.",
    "origin": "seed",
    "family": "role",
    "band": "destructive",
    "confirmed": false,
    "dq": {
     "mean": -0.3343,
     "lo": -0.4764,
     "hi": -0.2,
     "n": 50
    },
    "dd": {
     "mean": 0.2313,
     "lo": 0.1735,
     "hi": 0.2891,
     "n": 49
    }
   },
   {
    "id": "OB-016",
    "text": "Turn one document's tool into another document's question.",
    "origin": "seed",
    "family": "role",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.0497,
     "lo": -0.2212,
     "hi": 0.1133,
     "n": 50
    },
    "dd": {
     "mean": 0.055,
     "lo": 0.0133,
     "hi": 0.0967,
     "n": 50
    }
   },
   {
    "id": "OB-017",
    "text": "Only use data that already exists.",
    "origin": "seed",
    "family": "constraint",
    "band": "destructive",
    "confirmed": false,
    "dq": {
     "mean": -0.369,
     "lo": -0.5122,
     "hi": -0.2143,
     "n": 50
    },
    "dd": {
     "mean": 0.1139,
     "lo": 0.0765,
     "hi": 0.1531,
     "n": 49
    }
   },
   {
    "id": "OB-018",
    "text": "Assume measuring anything is free.",
    "origin": "seed",
    "family": "constraint",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.1319,
     "lo": -0.2903,
     "hi": 0.0214,
     "n": 50
    },
    "dd": {
     "mean": 0.0782,
     "lo": 0.0323,
     "hi": 0.1241,
     "n": 49
    }
   },
   {
    "id": "OB-019",
    "text": "The whole idea must fit in one experiment.",
    "origin": "seed",
    "family": "constraint",
    "band": "destructive",
    "confirmed": false,
    "dq": {
     "mean": -0.2641,
     "lo": -0.405,
     "hi": -0.1235,
     "n": 50
    },
    "dd": {
     "mean": 0.1156,
     "lo": 0.0731,
     "hi": 0.1599,
     "n": 49
    }
   },
   {
    "id": "OB-020",
    "text": "No new model may be trained.",
    "origin": "seed",
    "family": "constraint",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.1439,
     "lo": -0.3077,
     "hi": 0.0122,
     "n": 50
    },
    "dd": {
     "mean": 0.1003,
     "lo": 0.0544,
     "hi": 0.148,
     "n": 49
    }
   },
   {
    "id": "OB-021",
    "text": "State the bet without naming any technique.",
    "origin": "seed",
    "family": "ambiguity_specificity",
    "band": "destructive",
    "confirmed": false,
    "dq": {
     "mean": -0.2383,
     "lo": -0.388,
     "hi": -0.089,
     "n": 50
    },
    "dd": {
     "mean": 0.0817,
     "lo": 0.0317,
     "hi": 0.13,
     "n": 50
    }
   },
   {
    "id": "OB-022",
    "text": "Name the one number the whole idea depends on.",
    "origin": "seed",
    "family": "ambiguity_specificity",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.1507,
     "lo": -0.2762,
     "hi": -0.0244,
     "n": 50
    },
    "dd": {
     "mean": 0.055,
     "lo": 0.0083,
     "hi": 0.0967,
     "n": 50
    }
   },
   {
    "id": "OB-023",
    "text": "Define the key term so two documents disagree about it.",
    "origin": "seed",
    "family": "ambiguity_specificity",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.0466,
     "lo": -0.21,
     "hi": 0.1253,
     "n": 50
    },
    "dd": {
     "mean": 0.0714,
     "lo": 0.0221,
     "hi": 0.1207,
     "n": 49
    }
   },
   {
    "id": "OB-024",
    "text": "Leave one term undefined and build around it.",
    "origin": "seed",
    "family": "ambiguity_specificity",
    "band": "destructive",
    "confirmed": false,
    "dq": {
     "mean": -0.2038,
     "lo": -0.3525,
     "hi": -0.0537,
     "n": 50
    },
    "dd": {
     "mean": 0.0527,
     "lo": 0.0085,
     "hi": 0.0935,
     "n": 49
    }
   },
   {
    "id": "OB-025",
    "text": "Start from the result no one reported.",
    "origin": "seed",
    "family": "error",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.0333,
     "lo": -0.2115,
     "hi": 0.1524,
     "n": 50
    },
    "dd": {
     "mean": 0.0567,
     "lo": 0.0067,
     "hi": 0.1067,
     "n": 50
    }
   },
   {
    "id": "OB-026",
    "text": "Assume the strongest result in the documents is wrong.",
    "origin": "seed",
    "family": "error",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.0391,
     "lo": -0.2003,
     "hi": 0.1334,
     "n": 50
    },
    "dd": {
     "mean": 0.1767,
     "lo": 0.125,
     "hi": 0.2317,
     "n": 50
    }
   },
   {
    "id": "OB-027",
    "text": "Find where two documents fail for the same reason.",
    "origin": "seed",
    "family": "error",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.1093,
     "lo": -0.309,
     "hi": 0.1,
     "n": 50
    },
    "dd": {
     "mean": 0.1683,
     "lo": 0.1033,
     "hi": 0.23,
     "n": 50
    }
   },
   {
    "id": "OB-028",
    "text": "Move the question ten years into the future.",
    "origin": "seed",
    "family": "time",
    "band": "destructive",
    "confirmed": false,
    "dq": {
     "mean": -0.3617,
     "lo": -0.504,
     "hi": -0.2184,
     "n": 50
    },
    "dd": {
     "mean": 0.1528,
     "lo": 0.0868,
     "hi": 0.2188,
     "n": 48
    }
   },
   {
    "id": "OB-029",
    "text": "Ask what was knowable before any of these documents.",
    "origin": "seed",
    "family": "time",
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.1165,
     "lo": -0.2877,
     "hi": 0.0547,
     "n": 50
    },
    "dd": {
     "mean": 0.0884,
     "lo": 0.0323,
     "hi": 0.1463,
     "n": 49
    }
   },
   {
    "id": "OB-030",
    "text": "Look for the effect that appears only after the study ends.",
    "origin": "seed",
    "family": "time",
    "band": "destructive",
    "confirmed": false,
    "dq": {
     "mean": -0.2366,
     "lo": -0.4104,
     "hi": -0.0569,
     "n": 50
    },
    "dd": {
     "mean": 0.2602,
     "lo": 0.1972,
     "hi": 0.3214,
     "n": 49
    }
   },
   {
    "id": "E6-1",
    "text": "Begin where two documents cannot both be right.",
    "origin": "paper",
    "family": null,
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.0157,
     "lo": -0.1786,
     "hi": 0.1414,
     "n": 50
    },
    "dd": {
     "mean": 0.1167,
     "lo": 0.0667,
     "hi": 0.1667,
     "n": 50
    }
   },
   {
    "id": "E6-2",
    "text": "Begin from a limitation one source concedes or hides.",
    "origin": "paper",
    "family": null,
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.0246,
     "lo": -0.2034,
     "hi": 0.1663,
     "n": 50
    },
    "dd": {
     "mean": 0.0884,
     "lo": 0.0425,
     "hi": 0.1395,
     "n": 49
    }
   },
   {
    "id": "E6-3",
    "text": "Doubt anything new exists here; look for the counterexample.",
    "origin": "paper",
    "family": null,
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": 0.0112,
     "lo": -0.1613,
     "hi": 0.1848,
     "n": 50
    },
    "dd": {
     "mean": 0.2517,
     "lo": 0.1917,
     "hi": 0.305,
     "n": 50
    }
   },
   {
    "id": "E6-4",
    "text": "Find the premise every source takes for granted, then question it.",
    "origin": "paper",
    "family": null,
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": 0.0233,
     "lo": -0.152,
     "hi": 0.1999,
     "n": 50
    },
    "dd": {
     "mean": 0.2467,
     "lo": 0.1883,
     "hi": 0.305,
     "n": 50
    }
   },
   {
    "id": "E6-5",
    "text": "Begin with the least convincing claim in the documents.",
    "origin": "paper",
    "family": null,
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.0841,
     "lo": -0.2525,
     "hi": 0.0913,
     "n": 50
    },
    "dd": {
     "mean": 0.1718,
     "lo": 0.119,
     "hi": 0.2279,
     "n": 49
    }
   },
   {
    "id": "E6-6",
    "text": "Start from what the field needs that no document provides.",
    "origin": "paper",
    "family": null,
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": 0.0821,
     "lo": -0.0802,
     "hi": 0.2377,
     "n": 50
    },
    "dd": {
     "mean": 0.0816,
     "lo": 0.0289,
     "hi": 0.1344,
     "n": 49
    }
   },
   {
    "id": "CTRL-R-001",
    "text": "obligation",
    "origin": "control",
    "family": null,
    "band": "inert",
    "confirmed": false,
    "dq": {
     "mean": -0.0132,
     "lo": -0.1824,
     "hi": 0.1594,
     "n": 50
    },
    "dd": {
     "mean": 0.017,
     "lo": -0.0341,
     "hi": 0.068,
     "n": 49
    }
   },
   {
    "id": "CTRL-R-004",
    "text": "apartment",
    "origin": "control",
    "family": null,
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.0658,
     "lo": -0.2137,
     "hi": 0.085,
     "n": 50
    },
    "dd": {
     "mean": 0.0527,
     "lo": 0.0017,
     "hi": 0.1054,
     "n": 49
    }
   },
   {
    "id": "CTRL-R-014",
    "text": "migration",
    "origin": "control",
    "family": null,
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.2046,
     "lo": -0.3792,
     "hi": -0.0272,
     "n": 50
    },
    "dd": {
     "mean": 0.0493,
     "lo": 0.0102,
     "hi": 0.0919,
     "n": 49
    }
   },
   {
    "id": "CTRL-R-015",
    "text": "feedback",
    "origin": "control",
    "family": null,
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.1518,
     "lo": -0.31,
     "hi": 0.0094,
     "n": 50
    },
    "dd": {
     "mean": 0.0952,
     "lo": 0.0289,
     "hi": 0.1565,
     "n": 49
    }
   },
   {
    "id": "CTRL-R-020",
    "text": "tragedy",
    "origin": "control",
    "family": null,
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.0894,
     "lo": -0.2901,
     "hi": 0.1194,
     "n": 50
    },
    "dd": {
     "mean": 0.135,
     "lo": 0.08,
     "hi": 0.1917,
     "n": 50
    }
   },
   {
    "id": "CTRL-R-021",
    "text": "nose",
    "origin": "control",
    "family": null,
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": -0.1028,
     "lo": -0.2602,
     "hi": 0.0569,
     "n": 50
    },
    "dd": {
     "mean": 0.0816,
     "lo": 0.0238,
     "hi": 0.1395,
     "n": 49
    }
   },
   {
    "id": "CTRL-R-023",
    "text": "picture",
    "origin": "control",
    "family": null,
    "band": "inert",
    "confirmed": false,
    "dq": {
     "mean": -0.1004,
     "lo": -0.2657,
     "hi": 0.0566,
     "n": 50
    },
    "dd": {
     "mean": 0.0217,
     "lo": -0.02,
     "hi": 0.0667,
     "n": 50
    }
   },
   {
    "id": "CTRL-R-025",
    "text": "charm",
    "origin": "control",
    "family": null,
    "band": "inert",
    "confirmed": false,
    "dq": {
     "mean": -0.1554,
     "lo": -0.2674,
     "hi": -0.0406,
     "n": 50
    },
    "dd": {
     "mean": 0.0083,
     "lo": -0.045,
     "hi": 0.0617,
     "n": 50
    }
   },
   {
    "id": "CTRL-R-027",
    "text": "bust",
    "origin": "control",
    "family": null,
    "band": "liminal",
    "confirmed": false,
    "dq": {
     "mean": 0.0333,
     "lo": -0.13,
     "hi": 0.1945,
     "n": 50
    },
    "dd": {
     "mean": 0.0733,
     "lo": 0.0217,
     "hi": 0.1267,
     "n": 50
    }
   },
   {
    "id": "CTRL-R-030",
    "text": "closure",
    "origin": "control",
    "family": null,
    "band": "inert",
    "confirmed": false,
    "dq": {
     "mean": 0.0795,
     "lo": -0.0724,
     "hi": 0.2273,
     "n": 50
    },
    "dd": {
     "mean": 0.04,
     "lo": -0.0117,
     "hi": 0.09,
     "n": 50
    }
   },
   {
    "id": "C2-T",
    "text": null,
    "origin": "no_card",
    "family": null,
    "band": "inert",
    "confirmed": false,
    "dq": {
     "mean": -0.1262,
     "lo": -0.273,
     "hi": 0.0251,
     "n": 50
    },
    "dd": {
     "mean": 0.0119,
     "lo": -0.017,
     "hi": 0.0408,
     "n": 49
    }
   }
  ]
 }
};

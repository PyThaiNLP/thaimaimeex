# thaimaimee
Predict budget from project names of [ThaiME](http://nscr.nesdb.go.th/thaime-project/) 

We tried using `project_name` of ThaiME projects to predict how much `log_budget` they will get. The benchmark is done among LinearSVR, ULMFit and Multilingual Universal Sentence Encoder + LinearSVR. The metric is mean squared error (MSE).

| models       | mse      |
|--------------|----------|
| predict mean | 2.455477 |
| **LinearSVR**    | **1.166351** |
| ULMFit       | 1.182745 |
| USE          | 1.749290 |

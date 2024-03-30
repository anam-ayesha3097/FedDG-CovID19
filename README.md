# MOON+GA: Federated Domain Generalization Using Both Global and Local Adjustments
 Domain generalization has attracted significant attention from various disciplines, including Federated Learning(FL). 
 Methods that focus on this can be mainly divided into two directions: local and global model adjustment. Methods in 
 the local direction conduct the adjustment on local model training on the client side, while methods in the global 
 direction conduct the adjustment on the global model on the server side. Inspired by these two directions, we hypothesize
 that combining global and local adjustment methods could be beneficial to obtain more generalized results in the 
 centralized federated setting. To this end, we first review methods in the literature and implement Generalization 
 Adjustment [1] and MOON [2] as a global and local adjustment method, respectively. Then, we design a unified pipeline 
 to combine the MOON and GA and train the model on a federated domain generalization dataset(PACS [3]). 
 We quantitatively compare the results with baselines including FedAvg [4], FedProx [5], and SCAFFOLD [6] and show 
 that when combined with GA, the MOON approach could give 1% improvement and better performance in terms of generalization.

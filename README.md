# BlackboxAttack

Step 1. Run trainNN.py to create a target neural network classifier. This is the "oracle" that will be used as the target.

Step 2. Run trainSbst_NN_h1.py to train a substitute model. Parameters matter! Please help yourself tuning them to get a highest subtitute score.

Step 3. Run testSbst.py to craft and evaluate adversarial samples. FGS runs quite fast but OPT-L2 doesn't. As a reference, it took about 8 hours to run for 2,000 test images.

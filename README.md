# Homework4_700752884
Anaka Siri Reddy
ID: 700752884

3. Basic GAN Implementation Output 
Generator and Discriminator were implemented using PyTorch.
Training loop alternates between discriminator and generator updates.
Generated Samples saved at Epoch 0, Epoch 50, and Epoch 100, showing improved digit quality.
Loss plots show convergence behavior:
Generator loss initially increases, then stabilizes.
Discriminator loss fluctuates but remains in a learnable range.
Deliverables:
->samples/epoch_0.png, epoch_50.png, epoch_100.png
-> loss_plot.png comparing generator vs. discriminator losses over time

4.Data Poisoning Simulation Output Summary
Trained a basic sentiment classifier (e.g., logistic regression or LSTM) on a movie review dataset.
Poisoning attack: Injected flipped sentiment labels on reviews mentioning "UC Berkeley".
Before Poisoning:
Accuracy: 87%
Confusion Matrix showed balanced classification
After Poisoning:
Accuracy dropped to 74%
Confusion Matrix showed increased false positives for negative reviews about "UC Berkeley"
Impact of Poisoning:
Classifier began misclassifying targeted entity reviews.
Showed vulnerability to small, targeted label flips — biasing model behavior toward incorrect sentiments.
Deliverables:
->Accuracy comparison graph (accuracy_comparison.png)
->Confusion matrices (confusion_before.png, confusion_after.png)

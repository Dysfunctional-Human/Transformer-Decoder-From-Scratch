# Transformer-Decoder-From-Scratch

- dataset class - ready
- models - model initialization, model loading, model saving, forward functions, generate functions
- training script
- will add more stuff if need be

(Make it so in every run, we generate a prefix that is dd-mm-yy-hours-minutes-seconds). Folder structure should be trained_models -> bigram -> prefix_bigram -> prefix_bigram.pt, results.pt, plot
Model training pipeline -> train step -> test step -> Keep printing train and test loss at every epoch -> At every 1/10th epoch print a sample generation -> After every epoch check if test loss < max test loss, if yes then save the model, the results, and the loss curves

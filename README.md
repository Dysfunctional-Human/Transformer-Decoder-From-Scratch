# Transformer-Decoder-From-Scratch

- dataset class - ready
- models - model initialization, model loading, model saving, forward functions, generate functions  - ready
- training script - ready
- will add more stuff if need be

Done:
(Make it so in every run, we generate a prefix that is dd-mm-yy-hours-minutes-seconds). Folder structure should be trained_models -> bigram -> prefix_bigram -> prefix_bigram.pt, results.pt, plot
Model training pipeline -> train step -> test step -> Keep printing train and test loss at every epoch -> At every 1/10th epoch print a sample generation -> After every epoch check if test loss < max test loss, if yes then save the model, the results, and the loss curves

make config files rather than directly hardcoding in training pipeline
config_bigram.py
config_attention.py
config_transformer.py

Fix the float modulo bug before it causes silent issues during training - done
Add trained_models/ to .gitignore and remove the committed artifacts - done
Add requirements.txt
Switch save_results to JSON - done
Fix the CPU-safe state_dict saving  - done

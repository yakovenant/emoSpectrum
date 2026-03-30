import argparse
import torch
import pandas as pd
from train import ModelTrainer, model_evaluate
from nnets import make_model
from visualizer import plot_embeddings_with_dataloader

def model_test(args, test_loader, saved_models, er_obj, le_obj, device_test):
  for saved_model_name in saved_models:
    print(f"\nTesting model: {saved_model_name}")
    checkp = torch.load(saved_model_name)
    model_test = make_model(args)
    model_test.load_state_dict(checkp, strict=True)
    model_test.eval()
    print("Model tester init:")
    tester = ModelTrainer(model_test, args)
    _, test_acc_fin, test_f1_fin, test_rep_fin = model_evaluate(model_test, tester, test_loader, report_dict=True)
    print(f"\nFinal test results: \nTest Acc: {test_acc_fin:.4f} | Test F1: {test_f1_fin:.4f}")
    print(f"\nFinal test classification report:\n{pd.DataFrame(test_rep_fin).to_string(index=True)}\n")
    plot_embeddings_with_dataloader(model_test, test_loader, embedding_reducer=er_obj, label_encoder=le_obj)

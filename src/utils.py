import os
from huggingface_hub import HfApi

PRINT_LOG = []

def custom_print(text_message):
    print(text_message)
    if not isinstance(text_message, str): raise Exception("Wrong str data type.")
    PRINT_LOG.append(text_message)

def write_print_log(print_log_path):
    log_file = open(print_log_path, 'w')
    log_file.writelines(PRINT_LOG)
    log_file.close()


def download_data(repo_id, save_dir):
  api = HfApi()
  api.snapshot_download(
      repo_id=repo_id,
      #repo_type="dataset",
      local_dir=save_dir)


if __name__ == "__main__":
  
  data_source = "facebook/wav2vec2-base" # AbstractTTS/iemocap, xbgoose/dusha
  data_root = "/home/iakovenant/PycharmProjects/emoSpectrum/backbones" # "/media/ssd/datasets"
  print(f"Download {data_source} to {data_root}...")
  download_data(
      repo_id=data_source,
      save_dir=os.path.join(data_root, data_source))
  print("\nDONE!")

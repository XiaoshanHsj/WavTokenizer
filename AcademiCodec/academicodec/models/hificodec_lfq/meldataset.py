def get_dataset_filelist(a):
    with open(a.input_training_file, 'r') as f:
        training_files = ["/mnt/lynx4/users/cjs/dataset/LibriTTS_16khz/audio/"+l.strip() for l in f]
    with open(a.input_validation_file, 'r') as f:
        validation_files = ["/mnt/lynx4/users/cjs/dataset/LibriTTS_16khz/audio/"+l.strip() for l in f]
    return training_files, validation_files

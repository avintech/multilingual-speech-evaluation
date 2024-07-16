import multiprocessing
#from train import train
from preprocessing import pre_process
#from finetune_whisper import finetune

def run_pre_process(language):
    pre_process(language)

if __name__ == "__main__":
    #languages = ["chinese"]
    languages = ["chinese","tamil"]
    
    # Create a pool of processes
    with multiprocessing.Pool() as pool:
        # Map the function to the list of arguments
        pool.map(run_pre_process, languages)

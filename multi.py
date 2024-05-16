import multiprocessing
from train import train

def run_pre_process(language):
    train(language)

if __name__ == "__main__":
    languages = ["malay","chinese", "tamil"]
    
    # Create a pool of processes
    with multiprocessing.Pool() as pool:
        # Map the function to the list of arguments
        pool.map(run_pre_process, languages)

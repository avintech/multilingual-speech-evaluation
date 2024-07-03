import pkg_resources

libraries = [
    'joblib',
    'pandas',
    'sklearn',
    'streamlit',
    'streamlit-mic-recorder',
    'keras',
    'numpy',
    'librosa',
    'jieba',
    'sentence-transformers',
    'gpt4all'
]

for library in libraries:
    try:
        version = pkg_resources.get_distribution(library).version
        print(f"{library}=={version}")
    except pkg_resources.DistributionNotFound:
        print(f"{library} not found")

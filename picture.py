import pkg_resources

libraries = [
    'eng-to-ipa',
    'numpy',
    'parselmouth',
    'scipy',
    'whisper-timestamped',
    'dragonmapper',
    'torch',
    'torchaudio',
    'epitran',
    'librosa',
    'sklearn',
    'jiwer'
]
for library in libraries:
    try:
        version = pkg_resources.get_distribution(library.replace('-', '_')).version
        print(f"{library}=={version}")
    except pkg_resources.DistributionNotFound:
        print(f"{library} not found")

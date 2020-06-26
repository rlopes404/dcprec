dataset=['PinterestCore', 'RedditCore']

for d in dataset:
    for f in [25,50,75,100]:
        !python2 "main.py" --dataset=d --latent_dimension=f

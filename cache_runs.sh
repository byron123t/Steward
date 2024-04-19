for i in {1..10}
do
    python src/smart_runtime.py --url macys.com --task "find marriage registry with name JANE DOE" --verbose
    python src/smart_runtime.py --url shopping.google.com --task "Identify Nike Air Women's size 6 cross training shoes that offer free return." --verbose
    python src/smart_runtime.py --url united.com --task "Open the baggage fee calculator." --verbose
    python src/smart_runtime.py --url healthline.com --task "Browse a list of CBD product reviews." --verbose
    python src/smart_runtime.py --url imdb.com --task "Browse the list of top 250 movies and add the first one to my watchlist." --verbose
done

exec docker run -it --rm -p 3000:3000 \
$(docker images | awk '{print $1}' | awk 'NR==2'):$(docker images | awk '{print $2}' | awk 'NR==2') \
serve --production

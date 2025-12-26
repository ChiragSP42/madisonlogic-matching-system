# MadisonLogic Documentation

This is a POC that aims to build a matching system that will match company names to it's corresponding URL/domain from a database of the same. The deliverables according to the SOW include low latency (100k in 2 minutes), high accuracy (+85%).

## Setting up Meilisearch

Now this works for local setup or if you want to host it in EC2. If you're hosting it in EC2 you might have to change the url/port parameters. Ask AI for assistance

```bash
# Pull and run Meilisearch
docker run -d \
  --name meilisearch \
  -p 7700:7700 \
  -e MEILI_ENV='development' \
  -e MEILI_MASTER_KEY='testMasterKey123' \
  getmeili/meilisearch:latest
```

## Improving search capabilities of Meilisearch

So right now the script just includes the 'searchable attributes'. There are more configurations that need to be set up like 'Ranking rules' using the `update_ranking_rules` function, filterable parameters using the `update_filterable_attributes` function, sortable parameters using the `update_sortable_attributes` function.

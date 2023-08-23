/*
BigQuery SQL query to find all Swift files in GitHub repos.
It extracts:
- repo name
- ref (branch or tag)
- path
- file contents
- license
*/

WITH selected_repos as (
  SELECT
    f.id,
    f.repo_name as repo_name,
    f.ref as ref,
    f.path as path,
    l.license as license
  FROM
    `bigquery-public-data.github_repos.files` as f
    JOIN `bigquery-public-data.github_repos.licenses`
            as l on l.repo_name = f.repo_name
),
deduped_files as (
  SELECT
    f.id,
    MIN(f.repo_name) as repo_name,
    MIN(f.ref) as ref,
    MIN(f.path) as path,
    MIN(f.license) as license
  FROM
    selected_repos as f
  GROUP BY
    f.id
)
SELECT
  f.repo_name,
  f.ref,
  f.path,
  f.license,
  c.copies,
  c.content,
FROM
  deduped_files as f
  JOIN `bigquery-public-data.github_repos.contents` as c on f.id = c.id
WHERE
  NOT c.binary
  AND f.path like '%.swift'


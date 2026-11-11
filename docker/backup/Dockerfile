# Backup sidecar (spec 60 §6.6): scheduled Postgres base backup + logical dump, HelixDB volume tar,
# Garage critical buckets, retention pruning and a weekly self-check restore drill — inside compose,
# no host cron. Same major version as the `postgres` service so pg_basebackup/pg_restore match.
# Build context: docker/backup (keeps backups/ and target/ out of the context).
FROM postgres:17-alpine
RUN apk add --no-cache rclone curl tzdata
COPY *.sh /usr/local/lib/fa-backup/
RUN chmod 0755 /usr/local/lib/fa-backup/*.sh
ENV PATH=/usr/local/lib/fa-backup:$PATH
ENTRYPOINT ["/usr/local/lib/fa-backup/entrypoint.sh"]

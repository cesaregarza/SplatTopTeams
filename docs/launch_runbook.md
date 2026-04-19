# SplatTopTeams Launch Runbook

This is the first-launch path for `SplatTopTeams` on `teams-int.splat.top`.

## Deployment shape

- App repo: `SplatTopTeams`
- Config repo: `SplatTopConfig`
- Argo application: `splattop-teams-prod`
- Helm chart: `helm/splattop-teams`
- Release hostname: `teams-int.splat.top`
- Workloads render into namespace `default`

## Preconditions

Before the first sync:

- `SplatTopTeams` `main` is green on:
  - `uv run pytest -q`
  - `cd frontend && npm run build`
- `SplatTopConfig` validates the `splattop-teams` chart cleanly
- `db-secrets` exists in the target namespace and contains `RANKINGS_DATABASE_URL` or equivalent DB parts
- the image pull secret referenced by `global.imagePullSecrets` exists and can read from `registry.digitalocean.com/sendouq`

## First release

1. Merge the launch-ready app changes into `SplatTopTeams/main`.
2. Run the `Verify and Publish Images` workflow or wait for the push workflow to finish.
3. Copy the published tags from the workflow summary.
4. Update `SplatTopConfig/helm/splattop-teams/values-prod.yaml`:
   - `teamsApi.image.tag`
   - `teamsFrontend.image.tag`
   - `global.appImageTag` only if you want the same tag to remain the fallback default
5. Merge the config repo change.

## Sync and rollout

1. Sync the Argo app:

   ```bash
   argocd app sync splattop-teams-prod
   ```

2. Confirm Argo is healthy:

   ```bash
   argocd app get splattop-teams-prod
   ```

3. Confirm deployments rolled out:

   ```bash
   kubectl -n default rollout status deployment/splattop-teams-api
   kubectl -n default rollout status deployment/splattop-teams-frontend
   ```

4. Confirm the deployed image tags:

   ```bash
   kubectl -n default get deploy splattop-teams-api splattop-teams-frontend -o jsonpath='{range .items[*]}{.metadata.name}{" => "}{range .spec.template.spec.containers[*]}{.image}{" "}{end}{"\n"}{end}'
   ```

## Bootstrap the search snapshot

The site is not really usable until `team_search_refresh_runs` has at least one completed snapshot.

Run the refresh CronJob once immediately after the first deploy:

```bash
kubectl -n default create job --from=cronjob/splattop-teams-refresh splattop-teams-refresh-bootstrap-$(date +%s)
```

Then watch it:

```bash
kubectl -n default get jobs | grep splattop-teams-refresh-bootstrap
kubectl -n default logs job/<bootstrap-job-name>
```

## Post-sync checks

1. Health:

   ```bash
   curl -fsS https://teams-int.splat.top/api/health
   curl -fsS https://teams-int.splat.top/api/ready
   ```

2. Search:

   ```bash
   curl -fsS "https://teams-int.splat.top/api/team-search?q=healbook&top_n=5"
   ```

3. Frontend:

   - open `https://teams-int.splat.top`
   - confirm the app loads and the first search returns results

## Rollback

If the deploy is bad:

1. Revert the image tags in `SplatTopConfig/helm/splattop-teams/values-prod.yaml`
2. Merge the revert
3. Sync `splattop-teams-prod` again in Argo
4. Confirm the two deployments rolled back successfully with `kubectl rollout status`

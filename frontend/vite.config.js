import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  // Vite proxying is local-dev only. Deployed frontend traffic is handled by
  // nginx in dockerfiles/nginx.frontend.conf, so keep the local default pointed
  // at the local API unless a dev-specific override is explicitly provided.
  const proxyTarget = env.VITE_DEV_PROXY_API_TARGET || 'http://127.0.0.1:8000';
  const matchesProxyTarget = env.VITE_MATCHES_PROXY_TARGET || 'https://splat.top';

  return {
    server: {
      host: true,
      port: 3000,
      proxy: {
        ...(proxyTarget
          ? {
              '/api': {
                target: proxyTarget,
                changeOrigin: true,
                secure: true,
              },
            }
          : {}),
        '/__splat_top_api': {
          target: matchesProxyTarget,
          changeOrigin: true,
          secure: true,
          rewrite: (path) => path.replace(/^\/__splat_top_api/, ''),
        },
      },
    },
  };
});

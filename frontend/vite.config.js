import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const proxyTarget = env.VITE_PROXY_API_TARGET || '';
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

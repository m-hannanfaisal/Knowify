import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/admin/:path*",
        destination: "http://localhost:8000/admin/:path*",
      },
      {
        source: "/api/v1/:path*",
        destination: "http://localhost:8000/api/v1/:path*",
      },
      {
        source: "/metrics",
        destination: "http://localhost:8000/metrics",
      },
      {
        source: "/health",
        destination: "http://localhost:8000/health",
      },
    ];
  },
};

export default nextConfig;

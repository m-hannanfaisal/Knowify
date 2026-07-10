import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Knowify - Operator Admin Console",
  description: "Aggregated metrics, orchestrator trace timelines, and limits adjustment console.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

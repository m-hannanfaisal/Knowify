import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Knowify - Grounded QA Assistant",
  description: "Chat with your knowledge base with verifiable sources.",
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

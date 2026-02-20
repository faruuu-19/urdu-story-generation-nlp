import type { Metadata, Viewport } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'

const _geist = Geist({ subsets: ["latin"] });
const _geistMono = Geist_Mono({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: 'ریختہ- Urdu Story Generator',
  description: 'اردو میں بچوں کے لیے جادوئی کہانیاں بنائیں - Generate magical children stories in Urdu using AI',
}

export const viewport: Viewport = {
  themeColor: '#7A2D4D',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="ur" dir="rtl">
      <body className="font-sans antialiased">
        {children}
        <Analytics />
      </body>
    </html>
  )
}

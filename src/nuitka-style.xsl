<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
      xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="html" indent="yes"/>
  <xsl:template match="/">
    <html>
      <head>
        <title>Nuitka Crash Report</title>
        <style>
          body { font-family: monospace; padding: 1em; background: #f9f9f9; }
          pre { background: #eee; padding: 1em; white-space: pre-wrap; }
        </style>
      </head>
      <body>
        <h1>Nuitka Crash Report</h1>
        <p><b>Status:</b> <xsl:value-of select="/nuitka-compilation-report/@completion"/></p>
        <p><b>Exit Message:</b> <xsl:value-of select="/nuitka-compilation-report/@exit_message"/></p>
        <h2>SCons Errors</h2>
        <xsl:for-each select="//scons_error_report">
          <h3>Command</h3>
          <pre><xsl:value-of select="command"/></pre>
          <h3>Stderr</h3>
          <pre><xsl:value-of select="stderr"/></pre>
        </xsl:for-each>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>

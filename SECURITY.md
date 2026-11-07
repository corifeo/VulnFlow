# Security policy

## Supported versions

vulnflow is pre-1.0. Security fixes go into the latest release only.

## Reporting a vulnerability

Please don't open a public issue for a security problem in vulnflow itself.

Report it privately through GitHub: on the repository's **Security** tab, choose **Report a vulnerability**. Include what you found, how to reproduce it, and the version or commit you tested.

You should get a reply within a week. Once a fix is released, the advisory is published with credit to you, unless you'd rather stay anonymous.

## Scope

In scope: the vulnflow code. For example, unsafe handling of downloaded data, writing files outside the path you gave, or leaking an API key.

Out of scope: the content of the public feeds vulnflow downloads (NVD, CISA KEV, EPSS, CWE, CAPEC, MSRC). Report errors in that data to the publisher of the feed.

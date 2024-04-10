How to troubleshoot network issues?
-----------------------------------

- **I can't access resources from the Internet as I am behind a proxy**

  All network accesses are made with the `requests
  <https://requests.readthedocs.io/>`_ library. Usually, if your
  environment variables are properly set, you should not encounter any particular
  proxy issue. However, there are always situations where it may help to manually
  force the proxy settings in the configuration file.

  Edit your configuration file and add the following section. You may find where
  it is located with the ``traffic config -l`` command (in the shell) or in the
  ``traffic.config_file`` variable.

  .. parsed-literal::
      ## This sections contains all the specific options necessary to fit your
      ## network configuration (including proxy and ProxyCommand settings)
      [network]

      ## input here the arguments you need to pass as is to requests
      # http.proxy = http://proxy.company:8080
      # https.proxy = http://proxy.company:8080
      # http.proxy = socks5h://localhost:1234
      # https.proxy = socks5h://localhost:1234

How to troubleshoot network issues?
-----------------------------------

- **I can't access resources from the Internet as I am behind a proxy**

  All network accesses are made with the `requests
  <https://requests.readthedocs.io/>`_ library (or the `paramiko
  <http://www.paramiko.org/>`_ library for the Impala shell). Usually, if your
  environment variables are properly set, you should not encounter any particular
  proxy issue. However, there are always situations where it may help to manually
  force the proxy settings in the configuration file.
  
  Edit your configuration file and add the following section. You may find where
  it is located with the ``traffic config -l`` command (in the shell) or in the
  ``traffic.config_file`` variable.
  
  - The ``http.proxy`` options refers to usually HTTP and REST API calls;
  - The ``ssh.proxycommand`` option is used to proxy the SSH connexion through
    a specific ssh tunnel or through an http proxy (with ``connect.exe`` or
    ``nc -X connect``). According to your configuration, you may prefer to do
    the settings in your ``.ssh/config`` file.


  .. parsed-literal::
      ## This sections contains all the specific options necessary to fit your
      ## network configuration (including proxy and ProxyCommand settings)
      [network]
  
      ## input here the arguments you need to pass as is to requests
      # http.proxy = http://proxy.company:8080
      # https.proxy = http://proxy.company:8080
      # http.proxy = socks5h://localhost:1234
      # https.proxy = socks5h://localhost:1234
  
      ## input here the ProxyCommand you need to login to the Impala Shell
      ## WARNING:
      ##    Do not use %h and %p wildcards.
      ##    Write data.opensky-network.org and 2230 explicitly instead
      # ssh.proxycommand = ssh -W data.opensky-network.org:2230 proxy_ip:proxy_port
      # ssh.proxycommand = connect.exe -H proxy_ip:proxy_port data.opensky-network.org 2230
  
// Bicep template: single GPU spot VM that bootstraps the CGAL/Monty
// benchmark stack via cloud-init and runs the comparison matrix.
//
// Defaults to Standard_NC4as_T4_v3 (T4, 4 vCPU, 28 GB) Spot, which gives
// native amd64 + CUDA for habitat-sim at the cheapest GPU tier in Azure.
// The VM auto-deletes its OS disk on deletion.
//
// Usage:
//   az deployment group create \
//     -g <rg> -f main.bicep \
//     -p sshPublicKey="$(cat ~/.ssh/id_ed25519.pub)"

@description('Location for all resources.')
param location string = resourceGroup().location

@description('VM size. NC4as_T4_v3 is the cheapest CUDA GPU; D-series for CPU-only.')
@allowed([
  'Standard_NC4as_T4_v3'
  'Standard_NC8as_T4_v3'
  'Standard_NC6s_v3'
  'Standard_NV6ads_A10_v5'
  'Standard_NV12ads_A10_v5'
  'Standard_D4s_v5'
  'Standard_D8s_v5'
  'Standard_D16s_v5'
])
param vmSize string = 'Standard_NC4as_T4_v3'

@description('VM admin username.')
param adminUsername string = 'cgal'

@description('SSH public key (paste contents of ~/.ssh/id_*.pub).')
@secure()
param sshPublicKey string

@description('Use Spot pricing (much cheaper, can be evicted).')
param useSpot bool = true

@description('Maximum hourly price for Spot in USD; -1 means up to on-demand price.')
param spotMaxPrice int = -1

@description('Git ref of MCGPPeters/cgal to clone.')
param cgalRef string = 'main'

@description('Git ref of MCGPPeters/tbp.monty to clone.')
param montyRef string = 'cgal/main'

@description('SUITE env var passed to run_benchmark.sh.')
@allowed(['smoke', 'noise', 'full'])
param suite string = 'noise'

@description('Optional N_EVAL_EPOCHS shrink (empty = use experiment default).')
param nEvalEpochs string = ''

@description('Optional: storage account to receive results (logs + report).')
param resultsStorageAccount string = ''

@description('Optional: container in the storage account.')
param resultsContainer string = 'cgal-results'

var prefix = uniqueString(resourceGroup().id, deployment().name)
var vmName = 'cgal-bench-${take(prefix, 6)}'

// Cloud-init script: rendered into the VM customData.
var cloudInit = '''
#cloud-config
package_update: true
package_upgrade: false
packages:
  - git
  - curl
  - jq
  - ca-certificates
  - gnupg
  - lsb-release
write_files:
  - path: /etc/profile.d/cgal.sh
    content: |
      export CGAL_REF=${CGAL_REF}
      export MONTY_REF=${MONTY_REF}
      export SUITE=${SUITE}
      export N_EVAL_EPOCHS=${N_EVAL_EPOCHS}
      export RESULTS_SA=${RESULTS_SA}
      export RESULTS_CONTAINER=${RESULTS_CONTAINER}
  - path: /usr/local/bin/run-cgal-bench.sh
    permissions: '0755'
    content: |
      #!/usr/bin/env bash
      set -euxo pipefail
      source /etc/profile.d/cgal.sh
      cd /opt
      [ -d cgal ] || git clone --branch "${CGAL_REF}" --depth 1 https://github.com/MCGPPeters/cgal.git
      [ -d tbp.monty ] || git clone --branch "${MONTY_REF}" --depth 1 https://github.com/MCGPPeters/tbp.monty.git
      cd /opt/cgal/docker
      ./download_ycb.sh
      SUITE="${SUITE}" ./compose.sh up --build --abort-on-container-exit
      if [ -n "${RESULTS_SA}" ]; then
        az login --identity
        az storage container create --account-name "${RESULTS_SA}" --name "${RESULTS_CONTAINER}" --auth-mode login || true
        az storage blob upload-batch --account-name "${RESULTS_SA}" -d "${RESULTS_CONTAINER}/$(hostname)/$(date -u +%Y%m%dT%H%M%SZ)" -s /opt/cgal/docker/logs --auth-mode login
      fi
runcmd:
  # NVIDIA driver + container toolkit
  - distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  - curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  - curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  - apt-get update
  - DEBIAN_FRONTEND=noninteractive apt-get install -y ubuntu-drivers-common nvidia-container-toolkit
  - ubuntu-drivers autoinstall || true
  # Docker engine
  - install -m 0755 -d /etc/apt/keyrings
  - curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  - echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list
  - apt-get update
  - DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  - systemctl enable --now docker
  - usermod -aG docker ${ADMIN_USER}
  - nvidia-ctk runtime configure --runtime=docker
  - systemctl restart docker
  # az cli for blob upload (optional)
  - curl -sL https://aka.ms/InstallAzureCLIDeb | bash
  # Kick off the benchmark in the background so cloud-init returns
  - su - ${ADMIN_USER} -c "nohup /usr/local/bin/run-cgal-bench.sh > /var/log/cgal-bench.log 2>&1 &"
'''

var customData = base64(replace(replace(replace(replace(replace(replace(replace(cloudInit,
  '${CGAL_REF}', cgalRef),
  '${MONTY_REF}', montyRef),
  '${SUITE}', suite),
  '${N_EVAL_EPOCHS}', nEvalEpochs),
  '${RESULTS_SA}', resultsStorageAccount),
  '${RESULTS_CONTAINER}', resultsContainer),
  '${ADMIN_USER}', adminUsername))

// --------------------------------------------------------------------------
// Networking
// --------------------------------------------------------------------------

resource vnet 'Microsoft.Network/virtualNetworks@2024-01-01' = {
  name: '${vmName}-vnet'
  location: location
  properties: {
    addressSpace: { addressPrefixes: ['10.20.0.0/24'] }
    subnets: [{
      name: 'default'
      properties: { addressPrefix: '10.20.0.0/27' }
    }]
  }
}

resource pip 'Microsoft.Network/publicIPAddresses@2024-01-01' = {
  name: '${vmName}-pip'
  location: location
  sku: { name: 'Standard' }
  properties: {
    publicIPAllocationMethod: 'Static'
    dnsSettings: { domainNameLabel: vmName }
  }
}

resource nsg 'Microsoft.Network/networkSecurityGroups@2024-01-01' = {
  name: '${vmName}-nsg'
  location: location
  properties: {
    securityRules: [{
      name: 'allow-ssh'
      properties: {
        priority: 100
        direction: 'Inbound'
        access: 'Allow'
        protocol: 'Tcp'
        sourceAddressPrefix: '*'
        sourcePortRange: '*'
        destinationAddressPrefix: '*'
        destinationPortRange: '22'
      }
    }]
  }
}

resource nic 'Microsoft.Network/networkInterfaces@2024-01-01' = {
  name: '${vmName}-nic'
  location: location
  properties: {
    networkSecurityGroup: { id: nsg.id }
    ipConfigurations: [{
      name: 'ipconfig1'
      properties: {
        subnet: { id: vnet.properties.subnets[0].id }
        publicIPAddress: { id: pip.id }
        privateIPAllocationMethod: 'Dynamic'
      }
    }]
  }
}

// --------------------------------------------------------------------------
// VM
// --------------------------------------------------------------------------

resource vm 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: vmName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    hardwareProfile: { vmSize: vmSize }
    priority: useSpot ? 'Spot' : 'Regular'
    evictionPolicy: useSpot ? 'Delete' : null
    billingProfile: useSpot ? { maxPrice: spotMaxPrice } : null
    osProfile: {
      computerName: vmName
      adminUsername: adminUsername
      customData: customData
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: {
          publicKeys: [{
            path: '/home/${adminUsername}/.ssh/authorized_keys'
            keyData: sshPublicKey
          }]
        }
      }
    }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: 'ubuntu-24_04-lts'
        sku: 'server'
        version: 'latest'
      }
      osDisk: {
        createOption: 'FromImage'
        managedDisk: { storageAccountType: 'Premium_LRS' }
        deleteOption: 'Delete'
        diskSizeGB: 256
      }
    }
    networkProfile: {
      networkInterfaces: [{
        id: nic.id
        properties: { deleteOption: 'Delete' }
      }]
    }
  }
}

output sshCommand string = 'ssh ${adminUsername}@${pip.properties.dnsSettings.fqdn}'
output publicIp string = pip.properties.ipAddress
output fqdn string = pip.properties.dnsSettings.fqdn
output watchLog string = 'ssh ${adminUsername}@${pip.properties.dnsSettings.fqdn} tail -f /var/log/cgal-bench.log'
output fetchResults string = 'scp -r ${adminUsername}@${pip.properties.dnsSettings.fqdn}:/opt/cgal/docker/logs ./azure-results'

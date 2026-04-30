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

// Cloud-init script: loaded from external file to avoid Bicep interpolation
// of bash ${VAR} references inside multi-line strings.
var customData = base64(replace(replace(replace(replace(replace(replace(replace(loadTextContent('cloud-init.yaml'),
  '__CGAL_REF__', cgalRef),
  '__MONTY_REF__', montyRef),
  '__SUITE__', suite),
  '__N_EVAL_EPOCHS__', nEvalEpochs),
  '__RESULTS_SA__', resultsStorageAccount),
  '__RESULTS_CONTAINER__', resultsContainer),
  '__ADMIN_USER__', adminUsername))

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

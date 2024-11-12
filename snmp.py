from pysnmp.hlapi import *

def snmp_query(host, oid):
    for (errorIndication, errorStatus, errorIndex, varBinds) in getCmd(
            SnmpEngine(),
            CommunityData('public'),  # Use your community string
            UdpTransportTarget((host, 161)),
            ContextData(),
            ObjectType(ObjectIdentity(oid))):
        
        if errorIndication:
            print(errorIndication)
        elif errorStatus:
            print('%s at %s' % (errorStatus.prettyPrint(),
                                errorIndex and varBinds[int(errorIndex) - 1][0] or '?'))
        else:
            for varBind in varBinds:
                print(' = '.join([x.prettyPrint() for x in varBind]))

# Example OIDs for interface metrics
ifInOctets_oid = '1.3.6.1.2.1.2.2.1.10'  # Replace with your MIB OID for desired metrics
ifOutOctets_oid = '1.3.6.1.2.1.2.2.1.16'

host = '192.168.1.1'  # Replace with your device IP

print("Inbound Octets:")
snmp_query(host, ifInOctets_oid)

print("Outbound Octets:")
snmp_query(host, ifOutOctets_oid)

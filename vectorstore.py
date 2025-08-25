from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

def connect_astra(secure_connect_bundle: str, token: str):
    cloud = {"secure_connect_bundle": secure_connect_bundle}
    auth = PlainTextAuthProvider("token", token)
    cluster = Cluster(cloud=cloud, auth_provider=auth)
    session = cluster.connect()
    return cluster, session

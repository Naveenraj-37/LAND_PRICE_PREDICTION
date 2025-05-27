import oracledb

# Use a raw string to handle backslashes in the Windows path
oracledb.init_oracle_client(lib_dir=r"C:\Users\Moogambigai_M.Tech\land_price_prediction")

# Check if client libraries are loaded
client_version = oracledb.clientversion()
if client_version:
    print("Oracle Client version:", client_version)
else:
    print("No Oracle Client libraries found (running in thin mode).")

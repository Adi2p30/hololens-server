 * Running on http://127.0.0.1:8080
 * Running on http://192.168.4.133:8080

## Hololens Server and Edge Setup 
### Hololens Server (Deprecated)
The hololens server was the original setup for the project. It was used to run the server on the another device. It was made with the intention of recieving frames from the hololens.
 This setup is deprecated and has been replaced by the edge server setup.

## Edge Server
This server runs all 3 models simultaneously by enqueuing it in a queue and using multiprocessing from python to make a worker then take this and either run it parallelly or in a queue based on processing power and availability.

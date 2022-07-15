start calculation servers with
./start.sh ochem

If you have GPU card on your machine, start also:

./startgpu.sh gpu

It will start servers on card 0. 
You can add more servers on different cards by duplicating G0.xml to G1.xml, G2.xml ... Gn.xml,  
and explicitely changing the card number in Gn.xml

In the file "tasks" indicae number of servers simultaneously starting on one card/machine.
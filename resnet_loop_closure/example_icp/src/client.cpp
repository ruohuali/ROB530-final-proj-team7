// Client side C/C++ program to demonstrate Socket programming
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <vector>

std::vector<std::vector<unsigned char>> pseudoImage(int row_num, int col_num) {
    std::vector<std::vector<unsigned char>> arr;
    for (int i = 0; i < row_num; i++) {
        std::vector<unsigned char> row;
        for (int j = 0; j < col_num; j++) {
            row.push_back(i * col_num + j);
        }
        arr.push_back(row);
    }
    return arr;
}

void sendImage(std::vector<std::vector<unsigned char>> arr, int sock) {
    int row_num = arr.size();
    int col_num = arr[0].size();
	unsigned char msg[row_num*col_num] = {0};
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            msg[i*col_num + j] = arr[i][j];
        }
    }    
    send(sock, (void*)msg, row_num*col_num, 0);
    printf("all char sent\n");
}

std::vector<double> byteArray2Float(unsigned char* buffer) {

}

int main(int argc, char const *argv[])
{
    const int PORT = 8080;
    const char* HOST = "127.0.0.1";

    int sock = 0, valread;
    struct sockaddr_in serv_addr;
    double buffer[7] = {0};
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return -1;
    }
   
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    std::cout <<AF_INET<<std::endl;
    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0) 
    {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }
    std::cout << "Trying to connect..."<<std::endl;
    
    // Codes to Try out the port to connect
    /*int i = 0;
    for (i=0; i < 99999; i++){
    	serv_addr.sin_port = htons(i);
    	if(connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    		continue;
    	else{
    		std::cout<<"Connected!!"<<std::endl;
    		std::cout<<i<<std::endl;
    	}

    }
    std::cout<<"Tried range: [0, " << i << " Over"<<std::endl;
    return 0;*/
    if(connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
    	std::cout<<connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) <<std::endl;
        printf("\nConnection Failed \n");
        return -1;
    }
    std::cout <<"Connection Done"<<std::endl;
    std::vector<std::vector<unsigned char>> arr = pseudoImage(10, 20);
    
    int command;
    while (1){
    	std::cout << "Input your command: ";
		std::cin >> command;
		if (command == 1){
			sendImage(arr, sock);
			valread = read(sock , buffer, 1024);
			// printf("%s\n",buffer);    
			for (int i = 0; i < 7; i++) {
		    	printf("%f\n", buffer[i]);    
			}
			printf("all float received\n");
    	}
    	else
    		break;
    
    }
    
    
    return 0;
}

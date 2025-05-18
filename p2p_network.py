import socket
import threading
import pickle
import time
from queue import Queue

class P2PNode:
    """Handles peer-to-peer communication between nodes"""
    
    def __init__(self, node_id, host, port, neighbors=None):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.neighbors = neighbors or []
        self.message_queue = Queue()
        self.running = False
        self.message_handlers = {}
        
    def register_message_handler(self, message_type, handler_func):
        """Register function to handle specific message types"""
        self.message_handlers[message_type] = handler_func
        
    def start(self):
        """Start the communication threads"""
        self.running = True
        self.listen_thread = threading.Thread(target=self._listen)
        self.process_thread = threading.Thread(target=self._process_messages)
        
        self.listen_thread.daemon = True
        self.process_thread.daemon = True
        
        self.listen_thread.start()
        self.process_thread.start()
        
    def stop(self):
        """Stop all communication threads"""
        self.running = False
        if hasattr(self, 'listen_thread'):
            self.listen_thread.join(timeout=2.0)
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2.0)
            
    def _listen(self):
        """Listen for incoming connections"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(10)
        sock.settimeout(1.0)  # Allow for checking self.running
        
        while self.running:
            try:
                client, addr = sock.accept()
                # Handle in separate thread to avoid blocking
                threading.Thread(target=self._handle_client, args=(client,)).start()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Listen error: {e}")
                
        sock.close()
        
    def _handle_client(self, client_socket):
        """Process incoming message from another node"""
        try:
            # Receive message size first
            size_data = client_socket.recv(4)
            if not size_data:
                return
            msg_size = int.from_bytes(size_data, byteorder='big')
            
            # Receive the actual message
            chunks = []
            bytes_received = 0
            while bytes_received < msg_size:
                chunk = client_socket.recv(min(msg_size - bytes_received, 4096))
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_received += len(chunk)
                
            if bytes_received == msg_size:
                data = b''.join(chunks)
                message = pickle.loads(data)
                self.message_queue.put(message)
        except Exception as e:
            print(f"Client handling error: {e}")
        finally:
            client_socket.close()
            
    def _process_messages(self):
        """Process messages from the queue"""
        while self.running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get(block=False)
                    if message['type'] in self.message_handlers:
                        self.message_handlers[message['type']](message)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Message processing error: {e}")
                
    def send_message(self, peer_host, peer_port, message_type, **payload):
        """Send message to a specific peer"""
        message = {
            'type': message_type,
            'sender_id': self.node_id,
            'timestamp': time.time(),
            **payload
        }
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((peer_host, peer_port))
            
            # Serialize the message
            data = pickle.dumps(message)
            
            # Send size first
            size = len(data).to_bytes(4, byteorder='big')
            sock.sendall(size)
            
            # Then send the actual data
            sock.sendall(data)
            sock.close()
            return True
        except Exception as e:
            print(f"Error sending to {peer_host}:{peer_port}: {e}")
            return False
            
    def broadcast(self, message_type, **payload):
        """Send message to all neighbors"""
        success_count = 0
        for neighbor in self.neighbors:
            if self.send_message(neighbor['host'], neighbor['port'], message_type, **payload):
                success_count += 1
        return success_count
    
    def add_neighbor(self, node_id, host, port):
        """Add a new neighbor to the network"""
        for existing in self.neighbors:
            if existing['node_id'] == node_id:
                return False
                
        self.neighbors.append({
            'node_id': node_id,
            'host': host,
            'port': port
        })
        return True
        
    def remove_neighbor(self, node_id):
        """Remove a neighbor from the network"""
        self.neighbors = [n for n in self.neighbors if n['node_id'] != node_id]
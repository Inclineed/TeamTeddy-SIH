import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Mock chat sessions data
const mockChatSessions = [
  { id: 1, title: "Python Integration", lastMessage: "Working on Wails setup...", timestamp: "2 min ago", unread: 3 },
  { id: 2, title: "FastAPI Backend", lastMessage: "Server configuration complete", timestamp: "15 min ago", unread: 0 },
  { id: 3, title: "UI Components", lastMessage: "Building sidebar interface", timestamp: "1 hour ago", unread: 1 },
  { id: 4, title: "Database Setup", lastMessage: "MongoDB connection established", timestamp: "2 hours ago", unread: 0 },
  { id: 5, title: "Authentication", lastMessage: "JWT implementation in progress", timestamp: "1 day ago", unread: 0 },
  { id: 6, title: "File Upload", lastMessage: "Drag and drop functionality", timestamp: "2 days ago", unread: 0 }
];

// Mock chat messages data
const initialChatMessages = {
  1: [
    { id: 1, type: "assistant", content: "Hello! I'm here to help you with Python integration using Wails. What would you like to know?", timestamp: "10:30 AM" },
    { id: 2, type: "user", content: "How do I set up the Python backend with Wails?", timestamp: "10:32 AM" },
    { id: 3, type: "assistant", content: "Great question! First, you'll need to create a Python script in your backend directory. Then, you can call Python functions from your Go backend using the os/exec package. Would you like me to show you an example?", timestamp: "10:33 AM" },
    { id: 4, type: "user", content: "Yes, please show me an example!", timestamp: "10:35 AM" },
    { id: 5, type: "assistant", content: "Here's a simple example:\n\n```python\n# backend/python/hello.py\ndef greet(name):\n    return f'Hello, {name}!'\n```\n\nThen in your Go code, you can call it using subprocess execution.", timestamp: "10:36 AM" }
  ],
  2: [
    { id: 1, type: "assistant", content: "Welcome to FastAPI backend configuration! How can I assist you today?", timestamp: "9:15 AM" },
    { id: 2, type: "user", content: "I need help setting up the FastAPI server", timestamp: "9:17 AM" },
    { id: 3, type: "assistant", content: "Perfect! FastAPI is excellent for creating APIs quickly. You'll need to install FastAPI and uvicorn, then create your main.py file with route definitions.", timestamp: "9:18 AM" }
  ],
  3: [
    { id: 1, type: "assistant", content: "Let's work on your UI components! What kind of interface are you building?", timestamp: "8:45 AM" },
    { id: 2, type: "user", content: "I want to create a modern chat interface", timestamp: "8:47 AM" },
    { id: 3, type: "assistant", content: "Excellent choice! A chat interface with React and Framer Motion will be very smooth. We can use custom colors and animations to make it engaging.", timestamp: "8:48 AM" }
  ]
};

function MessageBubble({ message, index }) {
  const isUser = message.type === "user";
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.8 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ 
        duration: 0.4, 
        delay: index * 0.1,
        ease: [0.25, 0.46, 0.45, 0.94]
      }}
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        marginBottom: '16px',
        paddingLeft: isUser ? '60px' : '0',
        paddingRight: isUser ? '0' : '60px'
      }}
    >
      <div style={{
        maxWidth: '70%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start'
      }}>
        <motion.div
          whileHover={{ scale: 1.02 }}
          style={{
            padding: '12px 18px',
            borderRadius: isUser 
              ? '20px 20px 6px 20px' 
              : '20px 20px 20px 6px',
            backgroundColor: isUser 
              ? 'var(--white-smoke)' 
              : '#4A90E2',
            color: isUser 
              ? 'var(--outer-space)' 
              : 'var(--white-smoke)',
            boxShadow: isUser
              ? '0 4px 12px rgba(0, 0, 0, 0.15), 0 2px 4px rgba(0, 0, 0, 0.1)'
              : '0 4px 12px rgba(74, 144, 226, 0.25), 0 2px 4px rgba(74, 144, 226, 0.15)',
            background: isUser
              ? 'var(--white-smoke)'
              : 'linear-gradient(135deg, #4A90E2 0%, #357ABD 100%)',
            wordWrap: 'break-word',
            fontSize: '14px',
            lineHeight: '1.4'
          }}
        >
          {message.content.split('\n').map((line, i) => (
            <div key={i}>
              {line}
              {i < message.content.split('\n').length - 1 && <br />}
            </div>
          ))}
          
          {/* Streaming indicator */}
          {message.isStreaming && (
            <motion.span
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              style={{
                display: 'inline-block',
                marginLeft: '4px',
                color: isUser ? 'var(--outer-space)' : 'var(--white-smoke)',
                fontSize: '14px'
              }}
            >
              ▊
            </motion.span>
          )}
          
          {/* Display attached files */}
          {message.attachedFiles && message.attachedFiles.length > 0 && (
            <div style={{ marginTop: '8px' }}>
              {message.attachedFiles.map((file, fileIndex) => (
                <div
                  key={fileIndex}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '4px 8px',
                    backgroundColor: isUser ? 'rgba(55, 70, 80, 0.1)' : 'rgba(255, 255, 255, 0.2)',
                    borderRadius: '12px',
                    marginTop: fileIndex > 0 ? '4px' : '0',
                    fontSize: '12px',
                    fontWeight: '500'
                  }}
                >
                  <div style={{
                    width: '12px',
                    height: '12px',
                    backgroundColor: isUser ? 'var(--outer-space)' : 'var(--white-smoke)',
                    color: isUser ? 'var(--white-smoke)' : 'var(--outer-space)',
                    borderRadius: '2px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '8px',
                    fontWeight: 'bold'
                  }}>
                    PDF
                  </div>
                  <span style={{ fontSize: '11px' }}>{file.name}</span>
                </div>
              ))}
            </div>
          )}
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: (index * 0.1) + 0.3 }}
          style={{
            fontSize: '10px',
            color: 'rgba(241, 242, 238, 0.6)',
            marginTop: '4px',
            marginLeft: isUser ? '0' : '12px',
            marginRight: isUser ? '12px' : '0'
          }}
        >
          {message.timestamp}
        </motion.div>
      </div>
    </motion.div>
  );
}

function ChatWindow({ activeChat, chatMessages, onUpdateMessages, chatSessions }) {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ 
      behavior: "smooth",
      block: "end"
    });
  };

  useEffect(() => {
    // Load messages for active chat
    const currentChatMessages = chatMessages[activeChat] || [];
    setMessages(currentChatMessages);
  }, [activeChat, chatMessages]);

  useEffect(() => {
    // Auto-scroll when messages change
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if ((!newMessage.trim() && attachedFiles.length === 0) || isLoading) return;

    setIsLoading(true);
    const userMessage = {
      id: messages.length + 1,
      type: "user",
      content: newMessage || "📄 Attached files",
      attachedFiles: attachedFiles.map(f => ({ name: f.name, size: f.size })),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    onUpdateMessages(activeChat, updatedMessages);

    // Create assistant message placeholder for streaming
    const assistantMessage = {
      id: updatedMessages.length + 1,
      type: "assistant",
      content: "",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isStreaming: true
    };

    const messagesWithAssistant = [...updatedMessages, assistantMessage];
    setMessages(messagesWithAssistant);
    onUpdateMessages(activeChat, messagesWithAssistant);

    try {
      let uploadedFileName = null;
      
      // Step 1: Upload files if any are attached
      if (attachedFiles.length > 0) {
        const formData = new FormData();
        
        // Add attached files to FormData (using 'file' as the key based on API)
        attachedFiles.forEach((fileData) => {
          formData.append('file', fileData.file);
        });

        const uploadResponse = await fetch('http://localhost:8000/api/v1/documents/upload-and-process', {
          method: 'POST',
          body: formData,
        });

        if (!uploadResponse.ok) {
          throw new Error(`File upload failed: ${uploadResponse.status}`);
        }

        const uploadResult = await uploadResponse.json();
        console.log('Upload response:', uploadResult);
        
        // Extract the uploaded filename for RAG query
        uploadedFileName = uploadResult.upload?.saved_filename || uploadResult.upload?.original_filename;
        
        if (uploadResult.status === 'completed') {
          console.log(`✅ File processed successfully: ${uploadResult.indexing.indexed_chunks} chunks indexed`);
        }
      }

      // Step 2: Send question to RAG streaming endpoint
      const ragRequestBody = {
        question: newMessage || "What is this document about?",
        source_file: uploadedFileName || "string", // Use uploaded filename or default
        search_results: 5,
        temperature: 0.7
      };

      const ragResponse = await fetch('http://localhost:8000/api/v1/rag/ask-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: JSON.stringify(ragRequestBody)
      });

      if (!ragResponse.ok) {
        throw new Error(`RAG request failed: ${ragResponse.status}`);
      }

      // Handle streaming response
      const reader = ragResponse.body.getReader();
      const decoder = new TextDecoder();
      let streamedContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        
        // Handle different response formats (SSE or plain text)
        if (chunk.includes('data: ')) {
          // Server-Sent Events format
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data.trim() === '[DONE]') {
                break;
              }
              try {
                const parsed = JSON.parse(data);
                if (parsed.content) {
                  streamedContent += parsed.content;
                }
              } catch (parseError) {
                // If not JSON, treat as plain text
                streamedContent += data;
              }
            }
          }
        } else {
          // Plain text streaming
          streamedContent += chunk;
        }
        
        // Update the assistant message with streamed content
        const updatedAssistantMessage = {
          ...assistantMessage,
          content: streamedContent,
          isStreaming: true
        };
        
        const updatedMessagesWithStream = [
          ...updatedMessages,
          updatedAssistantMessage
        ];
        
        setMessages(updatedMessagesWithStream);
        onUpdateMessages(activeChat, updatedMessagesWithStream);
      }

      // Mark streaming as completed
      const finalAssistantMessage = {
        ...assistantMessage,
        content: streamedContent,
        isStreaming: false
      };
      
      const finalMessages = [
        ...updatedMessages,
        finalAssistantMessage
      ];
      
      setMessages(finalMessages);
      onUpdateMessages(activeChat, finalMessages);

    } catch (error) {
      console.error('Error in RAG API calls:', error);
      
      // Show error message to user
      const errorMessage = {
        ...assistantMessage,
        content: "Sorry, I encountered an error while processing your request. Please try again.",
        isStreaming: false
      };
      
      const errorMessages = [...updatedMessages, errorMessage];
      setMessages(errorMessages);
      onUpdateMessages(activeChat, errorMessages);
    } finally {
      setIsLoading(false);
      setNewMessage("");
      setAttachedFiles([]);
    }
  };

  // File handling functions
  const validateFile = (file) => {
    if (file.type !== 'application/pdf') {
      alert('Only PDF files are allowed');
      return false;
    }
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      alert('File size must be less than 10MB');
      return false;
    }
    return true;
  };

  const addAttachedFile = (file) => {
    if (!validateFile(file)) return;
    
    const fileData = {
      id: Date.now(),
      name: file.name,
      size: file.size,
      file: file
    };
    
    setAttachedFiles(prev => [...prev, fileData]);
  };

  const removeAttachedFile = (fileId) => {
    setAttachedFiles(prev => prev.filter(file => file.id !== fileId));
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragging(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    files.forEach(file => addAttachedFile(file));
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    files.forEach(file => addAttachedFile(file));
    e.target.value = ''; // Reset input
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const currentChat = chatSessions.find(chat => chat.id === activeChat);

  return (
    <div style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: 'rgba(55, 70, 80, 0.3)',
      borderRadius: '16px',
      border: '1px solid rgba(241, 242, 238, 0.1)',
      overflow: 'hidden',
      position: 'relative'
    }}>
      {/* Chat Header - Fixed */}
      <div style={{
        padding: '20px 24px',
        borderBottom: '1px solid rgba(241, 242, 238, 0.1)',
        backgroundColor: 'rgba(55, 70, 80, 0.5)',
        flexShrink: 0
      }}>
        <h2 style={{
          color: 'var(--mindaro)',
          fontSize: '18px',
          fontWeight: 'bold',
          margin: 0
        }}>
          {chatSessions.find(chat => chat.id === activeChat)?.title || 'Chat'}
        </h2>
        <p style={{
          color: 'rgba(241, 242, 238, 0.7)',
          fontSize: '12px',
          margin: '4px 0 0 0'
        }}>
          Active conversation
        </p>
      </div>

      {/* Messages Container - Scrollable Only */}
      <div 
        ref={chatContainerRef}
        className="chat-window-scrollbar"
        style={{
          flex: 1,
          overflowY: 'auto',
          overflowX: 'hidden',
          padding: '20px 24px',
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0
        }}
      >
        <AnimatePresence>
          {messages.map((message, index) => (
            <MessageBubble 
              key={`${message.id}-${activeChat}`} 
              message={message} 
              index={index}
            />
          ))}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Message Input - Fixed at Bottom */}
      <motion.form
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        onSubmit={handleSendMessage}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          padding: '20px 24px 24px 24px',
          borderTop: '1px solid rgba(241, 242, 238, 0.1)',
          backgroundColor: 'rgba(55, 70, 80, 0.5)',
          flexShrink: 0,
          position: 'relative'
        }}
      >
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          multiple
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />

        {/* Attached Files Display */}
        <AnimatePresence>
          {attachedFiles.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              style={{
                marginBottom: '12px',
                display: 'flex',
                flexWrap: 'wrap',
                gap: '8px'
              }}
            >
              {attachedFiles.map((file, index) => (
                <motion.div
                  key={file.id}
                  initial={{ opacity: 0, scale: 0.8, y: 20 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.8, y: -20 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  whileHover={{ scale: 1.02, y: -2 }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 12px',
                    backgroundColor: 'var(--mindaro)',
                    color: 'var(--outer-space)',
                    borderRadius: '20px',
                    boxShadow: '0 4px 12px rgba(220, 247, 99, 0.25), 0 2px 4px rgba(220, 247, 99, 0.15)',
                    fontSize: '12px',
                    fontWeight: '500',
                    maxWidth: '200px',
                    position: 'relative'
                  }}
                >
                  <div style={{
                    width: '16px',
                    height: '16px',
                    backgroundColor: 'var(--outer-space)',
                    borderRadius: '3px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '10px',
                    color: 'var(--white-smoke)',
                    fontWeight: 'bold'
                  }}>
                    PDF
                  </div>
                  <span style={{
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    flex: 1
                  }}>
                    {file.name}
                  </span>
                  <motion.button
                    type="button"
                    onClick={() => removeAttachedFile(file.id)}
                    whileHover={{ scale: 1.2, rotate: 90 }}
                    whileTap={{ scale: 0.9 }}
                    style={{
                      width: '16px',
                      height: '16px',
                      borderRadius: '50%',
                      backgroundColor: 'rgba(55, 70, 80, 0.2)',
                      border: 'none',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      fontSize: '12px',
                      color: 'var(--outer-space)',
                      fontWeight: 'bold',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    ×
                  </motion.button>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Drag Overlay */}
        <AnimatePresence>
          {isDragging && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: 'rgba(55, 70, 80, 0.8)',
                backdropFilter: 'blur(8px)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 10,
                borderRadius: '12px',
                border: '2px dashed var(--mindaro)'
              }}
            >
              <motion.div
                animate={{ 
                  scale: [1, 1.05, 1],
                  rotate: [0, 2, -2, 0]
                }}
                transition={{ 
                  duration: 2, 
                  repeat: Infinity,
                  ease: 'easeInOut' 
                }}
                style={{
                  textAlign: 'center',
                  color: 'var(--mindaro)',
                  fontSize: '18px',
                  fontWeight: '600'
                }}
              >
                📁 Drop your PDF file here
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        <div style={{
          display: 'flex',
          gap: '12px',
          alignItems: 'flex-end'
        }}>
          <motion.textarea
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            placeholder={isLoading ? "Waiting for response..." : "Type your message..."}
            rows={1}
            disabled={isLoading}
            style={{
              flex: 1,
              backgroundColor: isLoading ? 'rgba(241, 242, 238, 0.5)' : 'var(--white-smoke)',
              color: isLoading ? 'rgba(55, 70, 80, 0.5)' : 'var(--outer-space)',
              border: 'none',
              borderRadius: '12px',
              padding: '12px 16px',
              fontSize: '14px',
              resize: 'none',
              outline: 'none',
              fontFamily: 'inherit',
              height: '48px',
              maxHeight: '120px',
              cursor: isLoading ? 'not-allowed' : 'text',
              overflowY: 'auto'
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage(e);
              }
            }}
          />
          
          {/* File Attachment Button */}
          <motion.button
            type="button"
            onClick={triggerFileSelect}
            whileHover={{ 
              scale: 1.1, 
              backgroundColor: 'rgba(220, 247, 99, 0.1)',
              rotate: 15
            }}
            whileTap={{ scale: 0.9, rotate: -15 }}
            disabled={isLoading}
            style={{
              backgroundColor: 'transparent',
              border: '2px solid var(--mindaro)',
              borderRadius: '12px',
              width: '48px',
              height: '48px',
              cursor: isLoading ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--mindaro)',
              fontSize: '18px',
              transition: 'all 0.2s ease',
              opacity: isLoading ? 0.5 : 1
            }}
            title="Attach PDF file"
          >
            📎
          </motion.button>
          
          <motion.button
            type="submit"
            whileHover={!isLoading && (newMessage.trim() || attachedFiles.length > 0) ? { 
              scale: 1.1, 
              backgroundColor: 'rgba(220, 247, 99, 0.1)',
              rotate: 15 
            } : {}}
            whileTap={!isLoading && (newMessage.trim() || attachedFiles.length > 0) ? { scale: 0.9, rotate: -15 } : {}}
            disabled={(!newMessage.trim() && attachedFiles.length === 0) || isLoading}
            style={{
              backgroundColor: isLoading ? 'rgba(220, 247, 99, 0.5)' : ((newMessage.trim() || attachedFiles.length > 0) ? 'var(--mindaro)' : 'transparent'),
              color: isLoading ? 'rgba(55, 70, 80, 0.7)' : ((newMessage.trim() || attachedFiles.length > 0) ? 'var(--outer-space)' : 'var(--mindaro)'),
              border: isLoading ? 'none' : ((newMessage.trim() || attachedFiles.length > 0) ? 'none' : '2px solid var(--mindaro)'),
              borderRadius: '12px',
              width: '48px',
              height: '48px',
              cursor: ((newMessage.trim() || attachedFiles.length > 0) && !isLoading) ? 'pointer' : 'not-allowed',
              transition: 'all 0.2s ease',
              position: 'relative',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '18px',
              opacity: isLoading ? 0.5 : 1
            }}
            title={isLoading ? 'Sending...' : 'Send message'}
          >
            {isLoading ? (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                style={{
                  width: '16px',
                  height: '16px',
                  border: '2px solid transparent',
                  borderTop: '2px solid currentColor',
                  borderRadius: '50%'
                }}
              />
            ) : (
              '➤'
            )}
          </motion.button>
        </div>
      </motion.form>
    </div>
  );
}

function ChatSidebar({ isExpanded, onToggle, activeChat, onChatSelect, onNewChat, chatSessions }) {
  return (
    <motion.div
      initial={{ x: -280 }}
      animate={{ x: 0 }}
      style={{
        position: 'fixed',
        top: '20px',
        left: '20px',
        height: 'calc(100vh - 40px)',
        backgroundColor: 'rgba(55, 70, 80, 0.95)',
        backdropFilter: 'blur(10px)',
        borderRadius: '16px',
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(241, 242, 238, 0.1)',
        zIndex: 1000,
        overflow: 'hidden'
      }}
    >
      <motion.div
        animate={{ width: isExpanded ? 280 : 60 }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        style={{ height: '100%', position: 'relative' }}
      >
        {/* Header */}
        <div style={{
          padding: '20px',
          borderBottom: '1px solid rgba(241, 242, 238, 0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: isExpanded ? 'space-between' : 'center',
          flexDirection: isExpanded ? 'row' : 'column',
          gap: isExpanded ? 0 : '10px'
        }}>
          <AnimatePresence>
            {isExpanded && (
              <motion.h2
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
                style={{
                  color: 'var(--mindaro)',
                  fontSize: '18px',
                  fontWeight: 'bold',
                  margin: 0
                }}
              >
                Chat Sessions
              </motion.h2>
            )}
          </AnimatePresence>
          
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            flexDirection: isExpanded ? 'row' : 'column'
          }}>
            
            <motion.button
              onClick={onToggle}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              style={{
                background: 'transparent',
                border: '1px solid rgba(241, 242, 238, 0.2)',
                borderRadius: '8px',
                width: '32px',
                height: '32px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--white-smoke)'
              }}
            >
              <motion.div
                animate={{ rotate: isExpanded ? 180 : 0 }}
                transition={{ duration: 0.3 }}
              >
                ☰
              </motion.div>
            </motion.button>
          </div>
        </div>

        {/* New Chat Button Section */}
        <div style={{
          padding: isExpanded ? '15px 20px' : '10px',
          display: 'flex',
          justifyContent: isExpanded ? 'stretch' : 'center'
        }}>
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            whileHover={{ 
              scale: isExpanded ? 1.02 : 1.1, 
              backgroundColor: 'rgba(220, 247, 99, 0.1)' 
            }}
            whileTap={{ scale: 0.95 }}
            onClick={onNewChat}
            style={{
              background: 'transparent',
              border: '2px solid var(--mindaro)',
              borderRadius: isExpanded ? '10px' : '50%',
              padding: isExpanded ? '10px 16px' : '10px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--mindaro)',
              fontSize: isExpanded ? '14px' : '16px',
              fontWeight: '600',
              transition: 'all 0.2s ease',
              width: isExpanded ? '100%' : '40px',
              height: isExpanded ? 'auto' : '40px',
              boxShadow: '0 2px 8px rgba(220, 247, 99, 0.15)'
            }}
          >
            {isExpanded ? '+ New Chat' : '+'}
          </motion.button>
        </div>

        {/* Chat Sessions List */}
        <div 
          className="chat-sessions-scrollbar"
          style={{
            padding: isExpanded ? '10px' : '10px 5px',
            height: 'calc(100% - 140px)',
            overflowY: 'auto',
            overflowX: 'hidden'
          }}
        >
          {chatSessions.map((chat, index) => (
            <motion.div
              key={chat.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              whileHover={{ 
                scale: isExpanded ? 1.02 : 1.1,
                y: -2
              }}
              onClick={() => onChatSelect(chat.id)}
              style={{
                padding: isExpanded ? '12px' : '8px',
                marginBottom: '8px',
                borderRadius: '12px',
                cursor: 'pointer',
                backgroundColor: activeChat === chat.id ? 'rgba(220, 247, 99, 0.15)' : 'transparent',
                border: activeChat === chat.id ? '1px solid var(--mindaro)' : '1px solid transparent',
                transition: 'all 0.2s ease',
                position: 'relative',
                display: 'flex',
                alignItems: isExpanded ? 'flex-start' : 'center',
                justifyContent: isExpanded ? 'flex-start' : 'center'
              }}
              onHoverStart={() => {}}
              onHoverEnd={() => {}}
            >
              {/* Chat Avatar/Icon */}
              <div style={{
                width: '36px',
                height: '36px',
                borderRadius: '50%',
                backgroundColor: activeChat === chat.id ? 'var(--mindaro)' : 'rgba(241, 242, 238, 0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: activeChat === chat.id ? 'var(--outer-space)' : 'var(--white-smoke)',
                fontSize: '16px',
                fontWeight: 'bold',
                flexShrink: 0
              }}>
                {chat.title.charAt(0)}
              </div>

              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: 'auto' }}
                    exit={{ opacity: 0, width: 0 }}
                    transition={{ duration: 0.2 }}
                    style={{
                      marginLeft: '12px',
                      overflow: 'hidden',
                      flex: 1
                    }}
                  >
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'flex-start',
                      marginBottom: '4px'
                    }}>
                      <h4 style={{
                        color: activeChat === chat.id ? 'var(--mindaro)' : 'var(--white-smoke)',
                        fontSize: '14px',
                        fontWeight: '600',
                        margin: 0,
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        maxWidth: '140px'
                      }}>
                        {chat.title}
                      </h4>
                      
                      {chat.unread > 0 && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          style={{
                            backgroundColor: 'var(--mindaro)',
                            color: 'var(--outer-space)',
                            borderRadius: '10px',
                            fontSize: '10px',
                            fontWeight: 'bold',
                            padding: '2px 6px',
                            minWidth: '16px',
                            textAlign: 'center'
                          }}
                        >
                          {chat.unread}
                        </motion.div>
                      )}
                    </div>
                    
                    <p style={{
                      color: 'rgba(241, 242, 238, 0.7)',
                      fontSize: '12px',
                      margin: '0 0 4px 0',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis'
                    }}>
                      {chat.lastMessage}
                    </p>
                    
                    <p style={{
                      color: 'rgba(241, 242, 238, 0.5)',
                      fontSize: '10px',
                      margin: 0
                    }}>
                      {chat.timestamp}
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>


      </motion.div>
    </motion.div>
  );
}

function App() {
  const [msg, setMsg] = useState("");
  const [sidebarExpanded, setSidebarExpanded] = useState(true);
  const [activeChat, setActiveChat] = useState(1);
  const [chatSessions, setChatSessions] = useState(mockChatSessions);
  const [allChatMessages, setAllChatMessages] = useState(initialChatMessages);

  const toggleSidebar = () => {
    setSidebarExpanded(!sidebarExpanded);
  };

  const selectChat = (chatId) => {
    setActiveChat(chatId);
    const selectedChat = chatSessions.find(chat => chat.id === chatId);
    setMsg(`Active chat: ${selectedChat?.title}`);
  };

  const createNewChat = () => {
    const newChatId = chatSessions.length > 0 ? Math.max(...chatSessions.map(chat => chat.id)) + 1 : 1;
    
    const newChat = {
      id: newChatId,
      title: `New Chat ${newChatId}`,
      lastMessage: "Start a conversation...",
      timestamp: "now",
      unread: 0
    };
    
    // Add empty message array for new chat
    setAllChatMessages(prev => ({
      ...prev,
      [newChatId]: [
        {
          id: 1,
          type: "assistant",
          content: "Hello! I'm ready to help you. What would you like to discuss?",
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]
    }));
    
    setChatSessions(prev => [newChat, ...prev]);
    setActiveChat(newChatId);
    setMsg(`Active chat: ${newChat.title}`);
  };

  const updateChatMessages = (chatId, messages) => {
    setAllChatMessages(prev => ({
      ...prev,
      [chatId]: messages
    }));
  };

  return (
    <div style={{ 
      display: 'flex',
      height: '100vh',
      backgroundColor: 'var(--outer-space)',
      overflow: 'hidden'
    }}>
      {/* Chat Sidebar */}
      <ChatSidebar 
        isExpanded={sidebarExpanded}
        onToggle={toggleSidebar}
        activeChat={activeChat}
        onChatSelect={selectChat}
        onNewChat={createNewChat}
        chatSessions={chatSessions}
      />

      {/* Main Content Area */}
      <motion.div 
        animate={{ 
          marginLeft: sidebarExpanded ? '320px' : '100px'
        }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        style={{ 
          flex: 1,
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          height: '100vh',
          overflow: 'hidden',
          boxSizing: 'border-box'
        }}
      >
        <motion.h1 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          style={{ 
            color: 'var(--mindaro)', 
            marginBottom: '20px',
            textAlign: 'center',
            fontSize: '24px',
            flexShrink: 0
          }}
        >
          TeamTeddy Chat
        </motion.h1>
        
        {/* Chat Window */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            minHeight: 0,
            overflow: 'hidden'
          }}
        >
          <ChatWindow 
            activeChat={activeChat} 
            chatMessages={allChatMessages}
            onUpdateMessages={updateChatMessages}
            chatSessions={chatSessions}
          />
        </motion.div>
      </motion.div>
    </div>
  );
}

export default App;

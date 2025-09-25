import { useState, useEffect, useRef, createContext, useContext } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Theme Context
const ThemeContext = createContext();

const themes = {
  light: {
    // Base colors
    backgroundPrimary: '#fafbff',
    backgroundSecondary: '#f1f4fd',
    backgroundTertiary: '#e8eef9',
    
    // Surface colors
    surface: '#ffffff',
    surfaceHover: '#f8faff',
    surfaceActive: '#eef4ff',
    
    // Text colors
    textPrimary: '#1e293b',
    textSecondary: '#475569',
    textTertiary: '#94a3b8',
    
    // Accent colors
    accentPrimary: '#2563eb',
    accentSecondary: '#3b82f6',
    accentTertiary: '#60a5fa',
    accentGradient: '#2563eb', // Solid blue instead of gradient
    
    // Semantic colors
    success: '#059669',
    warning: '#d97706',
    error: '#dc2626',
    info: '#0284c7',
    
    // Border and shadow
    borderColor: 'rgba(37, 99, 235, 0.08)',
    borderColorLight: 'rgba(37, 99, 235, 0.04)',
    shadowSm: '0 1px 2px rgba(37, 99, 235, 0.06)',
    shadowMd: '0 4px 6px rgba(37, 99, 235, 0.08)',
    shadowLg: '0 10px 15px rgba(37, 99, 235, 0.1)',
    shadowXl: '0 20px 25px rgba(37, 99, 235, 0.12)',
    
    // Message specific
    userMessageBg: '#eff6ff',
    userMessageText: '#1e293b',
    assistantMessageBg: '#2563eb', // Solid blue instead of gradient
    assistantMessageText: '#ffffff',
    
    // Input colors
    inputBg: '#ffffff',
    inputBorder: 'rgba(37, 99, 235, 0.12)',
    inputFocus: '#2563eb',
    
    // Overlay
    overlayBg: 'rgba(250, 251, 255, 0.95)',
    glassBg: 'rgba(250, 251, 255, 0.85)',
    
    // Code
    codeBg: '#f1f5f9',
    codeText: '#334155'
  },
  dark: {
    // Base colors
    backgroundPrimary: '#0c1220',
    backgroundSecondary: '#1e2a3a',
    backgroundTertiary: '#2d3b4f',
    
    // Surface colors
    surface: '#1e2a3a',
    surfaceHover: '#2d3b4f',
    surfaceActive: '#3c4c63',
    
    // Text colors
    textPrimary: '#f1f5f9',
    textSecondary: '#cbd5e1',
    textTertiary: '#94a3b8',
    
    // Accent colors
    accentPrimary: '#3b82f6',
    accentSecondary: '#60a5fa',
    accentTertiary: '#93c5fd',
    accentGradient: '#3b82f6', // Solid blue instead of gradient
    
    // Semantic colors
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#06b6d4',
    
    // Border and shadow
    borderColor: 'rgba(59, 130, 246, 0.15)',
    borderColorLight: 'rgba(59, 130, 246, 0.08)',
    shadowSm: '0 1px 2px rgba(0, 0, 0, 0.3)',
    shadowMd: '0 4px 6px rgba(0, 0, 0, 0.35)',
    shadowLg: '0 10px 15px rgba(0, 0, 0, 0.4)',
    shadowXl: '0 20px 25px rgba(0, 0, 0, 0.5)',
    
    // Message specific
    userMessageBg: '#1e3a5f',
    userMessageText: '#f1f5f9',
    assistantMessageBg: '#3b82f6', // Solid blue instead of gradient
    assistantMessageText: '#ffffff',
    
    // Input colors
    inputBg: '#1e2a3a',
    inputBorder: 'rgba(59, 130, 246, 0.2)',
    inputFocus: '#3b82f6',
    
    // Overlay
    overlayBg: 'rgba(12, 18, 32, 0.95)',
    glassBg: 'rgba(12, 18, 32, 0.8)',
    
    // Code
    codeBg: '#1e2a3a',
    codeText: '#e2e8f0'
  }
};

// Mock data remains the same
const mockChatSessions = [
  { id: 1, title: "Python Integration", lastMessage: "Working on Wails setup...", timestamp: "2 min ago", unread: 0 },
  { id: 2, title: "FastAPI Backend", lastMessage: "Server configuration complete", timestamp: "15 min ago", unread: 0 },
  { id: 3, title: "UI Components", lastMessage: "Building sidebar interface", timestamp: "1 hour ago", unread: 0 },
  { id: 4, title: "Database Setup", lastMessage: "MongoDB connection established", timestamp: "2 hours ago", unread: 0 },
  { id: 5, title: "Authentication", lastMessage: "JWT implementation in progress", timestamp: "1 day ago", unread: 0 },
  { id: 6, title: "File Upload", lastMessage: "Drag and drop functionality", timestamp: "2 days ago", unread: 0 }
];

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



// Message Bubble Component
function MessageBubble({ message, index }) {
  const { theme } = useContext(ThemeContext);
  const currentTheme = themes[theme];
  const isUser = message.type === "user";
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ 
        duration: 0.3, 
        delay: index * 0.05,
        ease: [0.4, 0, 0.2, 1]
      }}
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        marginBottom: '20px',
        paddingLeft: isUser ? '15%' : '0',
        paddingRight: isUser ? '0' : '15%'
      }}
    >
      <div style={{
        maxWidth: '75%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        gap: '8px'
      }}>
        {!isUser && (
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '50%',
            background: currentTheme.accentGradient,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '4px'
          }}>
            <span style={{ fontSize: '16px' }}>🤖</span>
          </div>
        )}
        
        <motion.div
          whileHover={{ scale: 1.01 }}
          style={{
            padding: '16px 20px',
            borderRadius: isUser 
              ? '24px 24px 6px 24px' 
              : '24px 24px 24px 6px',
            background: isUser 
              ? currentTheme.userMessageBg 
              : currentTheme.assistantMessageBg,
            color: isUser 
              ? currentTheme.userMessageText 
              : currentTheme.assistantMessageText,
            boxShadow: currentTheme.shadowMd,
            fontSize: '15px',
            lineHeight: '1.6',
            backdropFilter: 'blur(10px)',
            position: 'relative',
            overflow: 'hidden'
          }}
        >
          {/* Subtle gradient overlay for depth */}
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: isUser 
              ? 'linear-gradient(135deg, transparent 0%, rgba(255,255,255,0.05) 100%)'
              : 'linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 100%)',
            pointerEvents: 'none'
          }} />
          
          <div style={{ position: 'relative', zIndex: 1, textAlign: 'left' }}>
            {message.isThinking ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span>Thinking</span>
                <div style={{ display: 'flex', gap: '4px' }}>
                  {[0, 1, 2].map((i) => (
                    <motion.div
                      key={i}
                      animate={{ 
                        scale: [1, 1.3, 1],
                        opacity: [0.5, 1, 0.5]
                      }}
                      transition={{ 
                        duration: 1,
                        repeat: Infinity,
                        delay: i * 0.2,
                        ease: "easeInOut"
                      }}
                      style={{
                        width: '6px',
                        height: '6px',
                        borderRadius: '50%',
                        backgroundColor: 'currentColor'
                      }}
                    />
                  ))}
                </div>
              </div>
            ) : (
              <>
                {message.content.split('\n').map((line, i) => (
                  <div key={i} style={{ textAlign: 'left' }}>
                    {line}
                    {i < message.content.split('\n').length - 1 && <br />}
                  </div>
                ))}
                
                {message.isStreaming && !message.isThinking && (
                  <motion.span
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                    style={{
                      display: 'inline-block',
                      marginLeft: '4px',
                      fontSize: '14px'
                    }}
                  >
                    ▊
                  </motion.span>
                )}
              </>
            )}
          </div>
          
          {message.attachedFiles && message.attachedFiles.length > 0 && (
            <div style={{ marginTop: '12px', position: 'relative', zIndex: 1 }}>
              {message.attachedFiles.map((file, fileIndex) => (
                <motion.div
                  key={fileIndex}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.1 * fileIndex }}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 12px',
                    backgroundColor: 'rgba(255, 255, 255, 0.15)',
                    backdropFilter: 'blur(10px)',
                    borderRadius: '16px',
                    marginTop: fileIndex > 0 ? '6px' : '0',
                    marginRight: '6px',
                    fontSize: '13px'
                  }}
                >
                  <span>📄</span>
                  <span>{file.name}</span>
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: (index * 0.05) + 0.2 }}
          style={{
            fontSize: '11px',
            color: currentTheme.textTertiary,
            paddingLeft: isUser ? '0' : '8px',
            paddingRight: isUser ? '8px' : '0'
          }}
        >
          {message.timestamp}
        </motion.div>
      </div>
    </motion.div>
  );
}

// Chat Window Component
function ChatWindow({ activeChat, chatMessages, onUpdateMessages, chatSessions, onChatSelect }) {
  const { theme } = useContext(ThemeContext);
  const currentTheme = themes[theme];
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isUploadingFiles, setIsUploadingFiles] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState([]); // Now stores uploaded file info
  const [isDragging, setIsDragging] = useState(false);
  const [fileError, setFileError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState({});
  
  // Voice input states
  const [isListening, setIsListening] = useState(false);
  const [voiceError, setVoiceError] = useState(null);
  const [speechRecognition, setSpeechRecognition] = useState(null);
  
  // Search states
  const [isSearching, setIsSearching] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const searchInputRef = useRef(null);
  const [isAtBottom, setIsAtBottom] = useState(true);

  // Initialize speech recognition
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';
        
        recognition.onresult = (event) => {
          let finalTranscript = '';
          let interimTranscript = '';
          
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscript += transcript;
            } else {
              interimTranscript += transcript;
            }
          }
          
          if (finalTranscript) {
            setNewMessage(prev => prev + finalTranscript + ' ');
            if (textareaRef.current) {
              const target = textareaRef.current;
              // Always keep at 52px for voice input to match button height
              target.style.height = '52px';
              
              // Only expand if there are actual line breaks or very long content
              setTimeout(() => {
                const scrollHeight = target.scrollHeight;
                if (scrollHeight > 60) { // Only expand if significantly larger
                  const newHeight = Math.min(scrollHeight, 120);
                  target.style.height = newHeight + 'px';
                }
              }, 0);
            }
          }
        };
        
        recognition.onerror = (event) => {
          setVoiceError(`Speech recognition error: ${event.error}`);
          setIsListening(false);
        };
        
        recognition.onend = () => {
          setIsListening(false);
        };
        
        setSpeechRecognition(recognition);
      } else {
        setVoiceError('Speech recognition not supported in this browser');
      }
    }
  }, []);

  // Voice input functions
  const startListening = async () => {
    if (!speechRecognition) {
      setVoiceError('Speech recognition not available');
      return;
    }

    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
      setVoiceError(null);
      speechRecognition.start();
      setIsListening(true);
    } catch (error) {
      setVoiceError('Microphone permission denied');
    }
  };

  const stopListening = () => {
    if (speechRecognition && isListening) {
      speechRecognition.stop();
    }
    setIsListening(false);
  };



  // Improved scroll to bottom function
  const scrollToBottom = (force = false) => {
    if (chatContainerRef.current && (isAtBottom || force)) {
      const container = chatContainerRef.current;
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  // Track scroll position
  const handleScroll = () => {
    if (chatContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current;
      const atBottom = scrollHeight - scrollTop - clientHeight < 100;
      setIsAtBottom(atBottom);
    }
  };

  useEffect(() => {
    const currentChatMessages = chatMessages[activeChat] || [];
    setMessages(currentChatMessages);
    // Force scroll on chat change
    setTimeout(() => scrollToBottom(true), 100);
  }, [activeChat, chatMessages]);

  useEffect(() => {
    // Auto-scroll when new messages are added
    if (messages.length > 0) {
      setTimeout(() => scrollToBottom(), 100);
    }
  }, [messages.length]);

  useEffect(() => {
    // Scroll during streaming
    if (isLoading) {
      const interval = setInterval(() => scrollToBottom(), 100);
      return () => clearInterval(interval);
    }
  }, [isLoading]);

const handleSendMessage = async (e) => {
  e.preventDefault();
  
  // Check if we can send (not uploading files, not already loading, have content or completed files)
  const hasContent = newMessage.trim();
  const hasCompletedFiles = attachedFiles.some(f => f.status === 'completed');
  
  if ((!hasContent && !hasCompletedFiles) || isLoading || isUploadingFiles) return;

  const messageContent = newMessage;
  const completedFiles = attachedFiles.filter(f => f.status === 'completed');
  
  setNewMessage("");
  setAttachedFiles([]);
  setFileError(null);
  setUploadProgress({});
  
  // Stop voice recording if active
  if (isListening) {
    stopListening();
  }
  
  if (textareaRef.current) {
    textareaRef.current.style.height = '52px';
  }

  setIsLoading(true);
  const userMessage = {
    id: messages.length + 1,
    type: "user",
    content: messageContent || "📄 Attached files",
    attachedFiles: completedFiles.map(f => ({ name: f.name, size: f.size, processedChunks: f.processedChunks })),
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  };

  const updatedMessages = [...messages, userMessage];
  setMessages(updatedMessages);
  onUpdateMessages(activeChat, updatedMessages);

  const assistantMessage = {
    id: updatedMessages.length + 1,
    type: "assistant",
    content: "Thinking...",
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    isStreaming: true,
    isThinking: true
  };

  const messagesWithAssistant = [...updatedMessages, assistantMessage];
  setMessages(messagesWithAssistant);
  onUpdateMessages(activeChat, messagesWithAssistant);

  try {
    // Files are already uploaded, just get the filename for RAG query
    let uploadedFileName = null;
    if (completedFiles.length > 0) {
      // Use the first uploaded file for RAG query (can be enhanced for multiple files)
      uploadedFileName = completedFiles[0].uploadedFileName;
    }

    // Send question to RAG streaming endpoint
    const ragRequestBody = {
      question: messageContent || "What is this document about? Please provide a comprehensive summary.",
      source_file: uploadedFileName || "", // Use uploaded filename or default
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
      if (chunk.includes('data:')) {
        // Server-Sent Events format
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data:')) {
            const data = line.slice(6);
            if (data.trim() === '[DONE]') break;
            
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
        content: streamedContent || "Thinking...",
        isStreaming: true,
        isThinking: !streamedContent // Remove thinking state once we have content
      };

      const updatedMessagesWithStream = [...updatedMessages, updatedAssistantMessage];
      setMessages(updatedMessagesWithStream);
      onUpdateMessages(activeChat, updatedMessagesWithStream);

      // Auto-scroll during streaming
      setTimeout(() => scrollToBottom(), 50);
    }

    // Mark streaming as completed
    const finalAssistantMessage = {
      ...assistantMessage,
      content: streamedContent,
      isStreaming: false,
      isThinking: false
    };

    const finalMessages = [...updatedMessages, finalAssistantMessage];
    setMessages(finalMessages);
    onUpdateMessages(activeChat, finalMessages);

    // Final scroll to bottom
    setTimeout(() => scrollToBottom(), 100);

  } catch (error) {
    console.error('Error in RAG API calls:', error);
    
    // Show error message to user
    const errorMessage = {
      ...assistantMessage,
      content: "Sorry, I encountered an error while processing your request. Please try again.",
      isStreaming: false,
      isThinking: false
    };

    const errorMessages = [...updatedMessages, errorMessage];
    setMessages(errorMessages);
    onUpdateMessages(activeChat, errorMessages);

    // Scroll to bottom for error message
    setTimeout(() => scrollToBottom(), 100);
  } finally {
    setIsLoading(false);
  }
};


  const validateFile = (file) => {
    const allowedTypes = ['application/pdf', 'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/webm'];
    if (!allowedTypes.includes(file.type)) {
      setFileError('Only PDF and audio files are allowed.');
      return false;
    }
    setFileError(null);
    return true;
  };

  const uploadFileAutomatically = async (file) => {
    const fileId = Date.now() + Math.random();
    const isAudioFile = file.type.startsWith('audio/');
    
    // Add file to state with uploading status
    const fileData = {
      id: fileId,
      name: file.name,
      size: file.size,
      status: isAudioFile ? 'completed' : 'uploading',
      type: isAudioFile ? 'audio' : 'document',
      file: isAudioFile ? file : null,
      uploadedFileName: null,
      error: null
    };
    
    setAttachedFiles(prev => [...prev, fileData]);
    
    // For audio files, just add them as attachments without uploading
    if (isAudioFile) {
      return;
    }
    
    setUploadProgress(prev => ({ ...prev, [fileId]: 0 }));
    setIsUploadingFiles(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      // Simulate progress for visual feedback
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        setUploadProgress(prev => ({ ...prev, [fileId]: progress }));
      }, 200);
      
      const uploadResponse = await fetch('http://localhost:8000/api/v1/documents/upload-and-process', {
        method: 'POST',
        body: formData,
      });
      
      clearInterval(progressInterval);
      setUploadProgress(prev => ({ ...prev, [fileId]: 95 }));
      
      if (!uploadResponse.ok) {
        throw new Error(`File upload failed: ${uploadResponse.status}`);
      }
      
      const uploadResult = await uploadResponse.json();
      console.log('Upload response:', uploadResult);
      
      // Update file status to completed
      setAttachedFiles(prev => prev.map(f => 
        f.id === fileId ? {
          ...f,
          status: 'completed',
          uploadedFileName: uploadResult.upload?.saved_filename || uploadResult.upload?.original_filename,
          processedChunks: uploadResult.indexing?.indexed_chunks
        } : f
      ));
      
      setUploadProgress(prev => ({ ...prev, [fileId]: 100 }));
      
    } catch (error) {
      console.error('Error uploading file:', error);
      
      // Update file status to error
      setAttachedFiles(prev => prev.map(f => 
        f.id === fileId ? {
          ...f,
          status: 'error',
          error: error.message
        } : f
      ));
      
      setUploadProgress(prev => ({ ...prev, [fileId]: 0 }));
    } finally {
      // Check if all files are done uploading
      setAttachedFiles(prev => {
        const stillUploading = prev.some(f => f.status === 'uploading');
        if (!stillUploading) {
          setIsUploadingFiles(false);
        }
        return prev;
      });
    }
  };
  
  const addAttachedFile = (file) => {
    if (!validateFile(file)) return;
    
    // Automatically start upload process
    uploadFileAutomatically(file);
    setFileError(null);
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
    e.target.value = '';
  };

  // Search functionality
  const handleSearch = (query) => {
    setSearchQuery(query);
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    // Search through all chat messages
    const results = [];
    Object.entries(chatMessages).forEach(([chatId, messages]) => {
      const chatSession = chatSessions.find(session => session.id === parseInt(chatId));
      messages.forEach((message) => {
        if (message.content.toLowerCase().includes(query.toLowerCase())) {
          results.push({
            chatId: parseInt(chatId),
            chatTitle: chatSession?.title || `Chat ${chatId}`,
            messageId: message.id,
            message: message,
            snippet: message.content.substring(0, 100) + (message.content.length > 100 ? '...' : '')
          });
        }
      });
    });
    
    setSearchResults(results);
  };

  const toggleSearch = () => {
    setIsSearching(!isSearching);
    if (!isSearching) {
      setTimeout(() => searchInputRef.current?.focus(), 100);
    } else {
      setSearchQuery("");
      setSearchResults([]);
    }
  };

  const selectSearchResult = (result) => {
    // Switch to the chat containing the search result
    if (result.chatId !== activeChat) {
      onChatSelect(result.chatId);
    }
    // Close the search
    setIsSearching(false);
    setSearchQuery("");
    setSearchResults([]);
  };

  return (
    <div style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: currentTheme.backgroundPrimary,
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        style={{
          padding: '20px 24px',
          borderBottom: `1px solid ${currentTheme.borderColor}`,
          backgroundColor: currentTheme.surface,
          backdropFilter: 'blur(10px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          position: 'relative',
          zIndex: 10
        }}
      >
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            background: currentTheme.accentPrimary,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '20px'
          }}>
            🤖
          </div>
          <h1 style={{
            color: currentTheme.accentPrimary,
            fontSize: '24px',
            fontWeight: '700',
            margin: 0
          }}>
            Teddy Rag.AI
          </h1>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Search Input (when expanded) */}
          <AnimatePresence>
            {isSearching && (
              <motion.div
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 300, opacity: 1 }}
                exit={{ width: 0, opacity: 0 }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
                style={{ overflow: 'hidden' }}
              >
                <input
                  ref={searchInputRef}
                  type="text"
                  value={searchQuery}
                  onChange={(e) => handleSearch(e.target.value)}
                  placeholder="Search in conversations..."
                  style={{
                    width: '100%',
                    padding: '10px 16px',
                    borderRadius: '12px',
                    border: `1px solid ${currentTheme.inputBorder}`,
                    backgroundColor: currentTheme.inputBg,
                    color: currentTheme.textPrimary,
                    fontSize: '14px',
                    outline: 'none',
                    transition: 'border-color 0.2s ease',
                    boxSizing: 'border-box'
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = currentTheme.accentPrimary;
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = currentTheme.inputBorder;
                  }}
                />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Search Button */}
          <motion.button
            onClick={toggleSearch}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '12px',
              backgroundColor: isSearching ? currentTheme.accentPrimary : 'transparent',
              border: `1px solid ${isSearching ? currentTheme.accentPrimary : currentTheme.borderColor}`,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '18px',
              color: isSearching ? '#ffffff' : currentTheme.textSecondary,
              transition: 'all 0.2s ease',
              flexShrink: 0
            }}
          >
            {isSearching ? '✕' : '🔍'}
          </motion.button>
        </div>
      </motion.div>

      {/* Search Results Dropdown */}
      <AnimatePresence>
        {isSearching && searchResults.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            style={{
              position: 'absolute',
              top: '84px',
              right: '24px',
              width: '400px',
              maxHeight: '300px',
              backgroundColor: currentTheme.surface,
              borderRadius: '16px',
              boxShadow: currentTheme.shadowXl,
              border: `1px solid ${currentTheme.borderColor}`,
              zIndex: 1000,
              overflow: 'hidden'
            }}
          >
            <div
              className="custom-scrollbar"
              style={{
                maxHeight: '300px',
                overflowY: 'auto'
              }}
            >
              {searchResults.map((result, index) => (
                <motion.div
                  key={`${result.chatId}-${result.messageId}`}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  onClick={() => selectSearchResult(result)}
                  style={{
                    padding: '16px 20px',
                    borderBottom: index < searchResults.length - 1 ? `1px solid ${currentTheme.borderColor}` : 'none',
                    cursor: 'pointer',
                    backgroundColor: 'transparent',
                    transition: 'background-color 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = currentTheme.surfaceHover;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  <div style={{
                    fontSize: '12px',
                    color: currentTheme.accentPrimary,
                    fontWeight: '600',
                    marginBottom: '4px'
                  }}>
                    {result.chatTitle}
                  </div>
                  <div style={{
                    fontSize: '14px',
                    color: currentTheme.textPrimary,
                    lineHeight: '1.4'
                  }}>
                    {result.snippet}
                  </div>
                  <div style={{
                    fontSize: '11px',
                    color: currentTheme.textTertiary,
                    marginTop: '6px'
                  }}>
                    {result.message.type === 'user' ? 'You' : 'Assistant'} • {result.message.timestamp}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Scroll to bottom button */}
      <AnimatePresence>
        {!isAtBottom && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            onClick={() => scrollToBottom(true)}
            style={{
              position: 'absolute',
              bottom: '140px',
              right: '40px',
              width: '40px',
              height: '40px',
              borderRadius: '50%',
              backgroundColor: currentTheme.accentPrimary,
              color: '#ffffff',
              border: 'none',
              boxShadow: currentTheme.shadowLg,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 100
            }}
          >
            ↓
          </motion.button>
        )}
      </AnimatePresence>

      {/* Messages Container */}
      <div 
        ref={chatContainerRef}
        onScroll={handleScroll}
        className="custom-scrollbar"
        style={{
          flex: 1,
          overflowY: 'auto',
          overflowX: 'hidden',
          padding: '20px 20px 150px 20px',
          scrollBehavior: 'smooth'
        }}
      >
        <div style={{
          maxWidth: '900px',
          margin: '0 auto',
          width: '100%'
        }}>
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
      </div>

      {/* Gradient Overlay */}
      <div style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: '200px',
        background: `linear-gradient(transparent, ${currentTheme.backgroundPrimary} 50%)`,
        pointerEvents: 'none',
        zIndex: 50
      }} />

      {/* Message Input */}
      <motion.form
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        onSubmit={handleSendMessage}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          position: 'absolute',
          bottom: '24px',
          left: '24px',
          right: '24px',
          maxWidth: '800px',
          margin: '0 auto',
          backgroundColor: currentTheme.glassBg,
          backdropFilter: 'blur(20px)',
          borderRadius: '20px',
          padding: '20px',
          boxShadow: currentTheme.shadowXl,
          border: `1px solid ${currentTheme.borderColor}`,
          zIndex: 100
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,audio/*"
          multiple
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />

        <AnimatePresence>
          {(fileError || voiceError) && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              style={{
                marginBottom: '16px',
                padding: '12px 16px',
                backgroundColor: `${currentTheme.error}20`,
                borderRadius: '12px',
                color: currentTheme.error,
                fontSize: '14px'
              }}
            >
              {fileError || voiceError}
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {attachedFiles.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                marginBottom: '16px',
                display: 'flex',
                flexDirection: 'column',
                gap: '10px'
              }}
            >
              {attachedFiles.map((file) => (
                <motion.div
                  key={file.id}
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  exit={{ scale: 0.8 }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '12px 16px',
                    background: file.status === 'error' 
                      ? `${currentTheme.error}20` 
                      : file.status === 'completed' 
                        ? currentTheme.accentPrimary 
                        : currentTheme.surfaceHover,
                    color: file.status === 'error' 
                      ? currentTheme.error 
                      : file.status === 'completed' 
                        ? '#ffffff' 
                        : currentTheme.textPrimary,
                    borderRadius: '16px',
                    fontSize: '13px',
                    border: file.status === 'error' 
                      ? `1px solid ${currentTheme.error}40` 
                      : 'none',
                    position: 'relative',
                    overflow: 'hidden'
                  }}
                >
                  {/* Upload progress bar */}
                  {file.status === 'uploading' && (
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${uploadProgress[file.id] || 0}%` }}
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        height: '100%',
                        background: `${currentTheme.accentPrimary}30`,
                        zIndex: 0
                      }}
                    />
                  )}
                  
                  <div style={{ position: 'relative', zIndex: 1, display: 'flex', alignItems: 'center', gap: '8px', flex: 1 }}>
                    <span>
                      {file.status === 'uploading' && (
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
                      )}
                      {file.status === 'completed' && (file.type === 'audio' ? '🎵' : '📄')}
                      {file.status === 'error' && '❌'}
                    </span>
                    
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: '500' }}>{file.name}</div>
                      {file.status === 'uploading' && (
                        <div style={{ fontSize: '11px', opacity: 0.8, marginTop: '2px' }}>
                          Uploading and processing...
                        </div>
                      )}
                      {file.status === 'completed' && file.processedChunks && (
                        <div style={{ fontSize: '11px', opacity: 0.8, marginTop: '2px' }}>
                          Processed {file.processedChunks} chunks
                        </div>
                      )}
                      {file.status === 'error' && (
                        <div style={{ fontSize: '11px', marginTop: '2px' }}>
                          {file.error || 'Upload failed'}
                        </div>
                      )}
                    </div>
                    
                    {file.status !== 'uploading' && (
                      <button
                        type="button"
                        onClick={() => removeAttachedFile(file.id)}
                        style={{
                          background: 'rgba(255, 255, 255, 0.2)',
                          border: 'none',
                          borderRadius: '50%',
                          width: '24px',
                          height: '24px',
                          cursor: 'pointer',
                          color: 'currentColor',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '16px'
                        }}
                      >
                        ×
                      </button>
                    )}
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {isDragging && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: currentTheme.overlayBg,
                borderRadius: '20px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: `2px dashed ${currentTheme.accentPrimary}`,
                zIndex: 10
              }}
            >
              <div style={{
                textAlign: 'center',
                color: currentTheme.accentPrimary
              }}>
                <div style={{ fontSize: '24px', marginBottom: '8px' }}>📁</div>
                <div>Drop PDF or audio files here</div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div style={{ 
          display: 'flex', 
          gap: '12px', 
          alignItems: 'center',
          minHeight: '52px'
        }}>
          {/* Attachment button - Left */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading || isListening}
            style={{
              width: '52px',
              height: '52px',
              borderRadius: '16px',
              backgroundColor: 'transparent',
              border: `2px solid ${currentTheme.borderColor}`,
              cursor: (isLoading || isListening) ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '20px',
              color: currentTheme.textSecondary,
              transition: 'all 0.2s ease',
              opacity: (isLoading || isListening) ? 0.5 : 1,
              flexShrink: 0
            }}
          >
            📎
          </button>
          
          {/* Text area - Center (expandable) */}
          <textarea
            ref={textareaRef}
            value={newMessage}
            onChange={(e) => {
              setNewMessage(e.target.value);
              const target = e.target;
              
              // Always reset to button height first
              target.style.height = '52px';
              
              // Only expand if content contains line breaks OR is very long
              const hasLineBreaks = e.target.value.includes('\n');
              const isVeryLong = e.target.value.length > 100; // Adjust threshold as needed
              
              if (hasLineBreaks || isVeryLong) {
                // Use setTimeout to ensure accurate scrollHeight calculation
                setTimeout(() => {
                  const scrollHeight = target.scrollHeight;
                  if (scrollHeight > 60) { // Only if actually needs more space
                    const newHeight = Math.min(scrollHeight, 120);
                    target.style.height = newHeight + 'px';
                  }
                }, 0);
              }
            }}
            placeholder={isListening ? "Listening..." : "Type your message..."}
            className="custom-scrollbar-compact"
            style={{
              flex: 1,
              backgroundColor: currentTheme.inputBg,
              color: currentTheme.textPrimary,
              border: `1px solid ${isListening ? currentTheme.accentPrimary : currentTheme.inputBorder}`,
              borderRadius: '16px',
              padding: '15px 18px', // Adjusted to match button height exactly
              fontSize: '15px',
              resize: 'none',
              outline: 'none',
              fontFamily: 'inherit',
              height: '52px',
              minHeight: '52px',
              maxHeight: '120px',
              transition: 'height 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease',
              boxShadow: isListening ? `0 0 0 2px ${currentTheme.accentPrimary}20` : 'none',
              lineHeight: '20px', // Adjusted for better alignment
              verticalAlign: 'top',
              boxSizing: 'border-box' // Ensure consistent sizing
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage(e);
              }
            }}
          />
          
          {/* Voice input button - Right */}
          <motion.button
            type="button"
            onClick={isListening ? stopListening : startListening}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            style={{
              width: '52px',
              height: '52px',
              borderRadius: '16px',
              backgroundColor: isListening ? currentTheme.accentPrimary : 'transparent',
              border: `2px solid ${isListening ? currentTheme.accentPrimary : currentTheme.borderColor}`,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '20px',
              color: isListening ? '#ffffff' : currentTheme.textSecondary,
              transition: 'all 0.3s ease',
              opacity: 1,
              flexShrink: 0
            }}
          >
            {isListening ? (
              <motion.div
                animate={{ 
                  scale: [1, 1.3, 1],
                  rotate: [0, 5, -5, 0]
                }}
                transition={{ 
                  duration: 0.8, 
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              >
                🎤
              </motion.div>
            ) : (
              '🎤'
            )}
          </motion.button>
          
          {/* Send button - Right */}
          <motion.button
            type="submit"
            whileHover={(!isLoading && !isUploadingFiles) ? { scale: 1.05 } : {}}
            whileTap={(!isLoading && !isUploadingFiles) ? { scale: 0.95 } : {}}
            disabled={isLoading || isUploadingFiles}
            style={{
              width: '52px',
              height: '52px',
              borderRadius: '16px',
              background: (isLoading || isUploadingFiles)
                ? currentTheme.surfaceHover
                : (newMessage.trim() || attachedFiles.some(f => f.status === 'completed')) 
                  ? currentTheme.accentPrimary 
                  : currentTheme.surfaceHover,
              border: 'none',
              cursor: ((newMessage.trim() || attachedFiles.some(f => f.status === 'completed')) && !isLoading && !isUploadingFiles) 
                ? 'pointer' 
                : 'not-allowed',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '20px',
              color: (isLoading || isUploadingFiles)
                ? currentTheme.textSecondary
                : (newMessage.trim() || attachedFiles.some(f => f.status === 'completed')) 
                  ? '#ffffff' 
                  : currentTheme.textSecondary,
              transition: 'all 0.2s ease',
              opacity: (isLoading || isUploadingFiles) ? 0.7 : 1,
              flexShrink: 0
            }}
          >
            {isLoading ? (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                style={{
                  width: '20px',
                  height: '20px',
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

// Theme Provider Component
function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('dark');
  
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
  }, []);
  
  useEffect(() => {
    localStorage.setItem('theme', theme);
    const currentTheme = themes[theme];
    
    // Apply CSS custom properties to document root
    Object.entries(currentTheme).forEach(([key, value]) => {
      document.documentElement.style.setProperty(`--${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`, value);
    });
  }, [theme]);
  
  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// Sidebar Component
function Sidebar({ activeChat, chatMessages, onUpdateMessages, chatSessions, onChatSelect, onNewChat }) {
  const { theme, setTheme } = useContext(ThemeContext);
  const currentTheme = themes[theme];
  const [sidebarExpanded, setSidebarExpanded] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  
  const toggleSidebar = () => {
    setSidebarExpanded(!sidebarExpanded);
  };
  
  return (
    <motion.div
      initial={{ x: -300 }}
      animate={{ x: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      style={{
        height: '100vh',
        backgroundColor: currentTheme.backgroundSecondary,
        borderRight: `1px solid ${currentTheme.borderColor}`,
        overflow: 'hidden',
        flexShrink: 0,
        position: 'relative',
        willChange: 'width'
      }}
    >
      <motion.div
        animate={{ width: sidebarExpanded ? 320 : 80 }}
        transition={{ 
          duration: 0.4, 
          ease: [0.23, 1, 0.320, 1], // Custom cubic-bezier for smoother animation
          layout: { duration: 0.4 }
        }}
        style={{ 
          height: '100%',
          willChange: 'width'
        }}
      >
        {/* Header */}
        <div style={{
          padding: '24px 20px',
          borderBottom: `1px solid ${currentTheme.borderColor}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: sidebarExpanded ? 'space-between' : 'center',
        }}>
          <AnimatePresence>
            {sidebarExpanded && (
              <motion.h2
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.25, delay: 0.15 }}
                style={{
                  color: currentTheme.accentPrimary,
                  fontSize: '20px',
                  fontWeight: 'bold',
                  margin: 0,
                  whiteSpace: 'nowrap'
                }}
              >
                Conversations
              </motion.h2>
            )}
          </AnimatePresence>
          
          <motion.button
            onClick={toggleSidebar}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            style={{
              background: 'transparent',
              border: `1px solid ${currentTheme.borderColor}`,
              borderRadius: '8px',
              width: '40px',
              height: '40px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: currentTheme.textPrimary,
              fontSize: '18px'
            }}
          >
            <motion.div
              animate={{ rotate: sidebarExpanded ? 0 : 180 }}
              transition={{ duration: 0.4, ease: [0.23, 1, 0.320, 1] }}
            >
              ☰
            </motion.div>
          </motion.button>
        </div>
        
        {/* New Chat Button */}
        <div style={{
          padding: sidebarExpanded ? '20px' : '20px 10px',
          borderBottom: `1px solid ${currentTheme.borderColor}`
        }}>
          <motion.button
            whileHover={{ scale: sidebarExpanded ? 1.02 : 1.1 }}
            whileTap={{ scale: 0.95 }}
            onClick={onNewChat}
            style={{
              width: '100%',
              padding: sidebarExpanded ? '16px 20px' : '16px 0',
              background: currentTheme.accentGradient,
              border: 'none',
              borderRadius: '16px',
              color: '#ffffff',
              fontSize: sidebarExpanded ? '16px' : '20px',
              fontWeight: '600',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              boxShadow: currentTheme.shadowMd
            }}
          >
            <span>+</span>
            {sidebarExpanded && <span>New Chat</span>}
          </motion.button>
        </div>
        
        {/* Chat Sessions */}
        <div 
          className="custom-scrollbar"
          style={{
            height: 'calc(100% - 280px)', // More space to prevent overlap with settings
            overflowY: 'auto',
            padding: sidebarExpanded ? '16px' : '16px 8px',
            marginBottom: '8px' // Additional margin
          }}
        >
          {chatSessions.map((chat, index) => (
            <motion.div
              key={chat.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              whileHover={{ scale: sidebarExpanded ? 1.02 : 1.1 }}
              onClick={() => onChatSelect(chat.id)}
              style={{
                padding: sidebarExpanded ? '16px' : '12px',
                marginBottom: '8px',
                borderRadius: '16px',
                cursor: 'pointer',
                backgroundColor: activeChat === chat.id 
                  ? currentTheme.surfaceHover 
                  : 'transparent',
                border: activeChat === chat.id 
                  ? `1px solid ${currentTheme.accentPrimary}` 
                  : 'none',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: sidebarExpanded ? '12px' : '0',
                justifyContent: sidebarExpanded ? 'flex-start' : 'center'
              }}
            >
              <div style={{
                width: '40px',
                height: '40px',
                borderRadius: '50%',
                background: currentTheme.accentGradient,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '16px',
                flexShrink: 0
              }}>
                {chat.title.charAt(0)}
              </div>
              
              <AnimatePresence>
                {sidebarExpanded && (
                  <motion.div
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: 'auto' }}
                    exit={{ opacity: 0, width: 0 }}
                    transition={{ 
                      duration: 0.3,
                      delay: sidebarExpanded ? 0.2 : 0,
                      ease: "easeOut"
                    }}
                    style={{ 
                      overflow: 'hidden', 
                      flex: 1,
                      whiteSpace: 'nowrap'
                    }}
                  >
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'flex-start',
                      marginBottom: '4px'
                    }}>
                      <h4 style={{
                        color: activeChat === chat.id 
                          ? currentTheme.accentPrimary 
                          : currentTheme.textPrimary,
                        fontSize: '14px',
                        fontWeight: '600',
                        margin: 0,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        maxWidth: '180px'
                      }}>
                        {chat.title}
                      </h4>
                      
                      {chat.unread > 0 && (
                        <div style={{
                          backgroundColor: currentTheme.accentPrimary,
                          color: '#ffffff',
                          borderRadius: '10px',
                          fontSize: '10px',
                          fontWeight: 'bold',
                          padding: '2px 6px',
                          minWidth: '16px',
                          textAlign: 'left'
                        }}>
                          {chat.unread}
                        </div>
                      )}
                    </div>
                    
                    <p style={{
                      color: currentTheme.textSecondary,
                      fontSize: '12px',
                      margin: '0 0 4px 0',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap'
                    }}>
                      {chat.lastMessage}
                    </p>
                    
                    <p style={{
                      color: currentTheme.textTertiary,
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

        {/* Settings Section */}
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          borderTop: `1px solid ${currentTheme.borderColor}`,
          backgroundColor: currentTheme.backgroundSecondary
        }}>
          {/* Settings Button */}
          <div style={{
            padding: sidebarExpanded ? '16px 20px' : '16px 10px'
          }}>
            <motion.button
              onClick={() => setShowSettings(!showSettings)}
              whileHover={{ scale: sidebarExpanded ? 1.02 : 1.1 }}
              whileTap={{ scale: 0.95 }}
              style={{
                width: '100%',
                padding: sidebarExpanded ? '12px 16px' : '12px 0',
                background: showSettings ? currentTheme.surfaceHover : 'transparent',
                border: `1px solid ${currentTheme.borderColor}`,
                borderRadius: '12px',
                color: currentTheme.textPrimary,
                fontSize: sidebarExpanded ? '14px' : '18px',
                fontWeight: '500',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: sidebarExpanded ? 'flex-start' : 'center',
                gap: sidebarExpanded ? '12px' : '0',
                transition: 'all 0.2s ease'
              }}
            >
              <span>⚙️</span>
              {sidebarExpanded && <span>Settings</span>}
            </motion.button>
          </div>

          {/* Settings Panel */}
          <AnimatePresence>
            {showSettings && sidebarExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3 }}
                style={{
                  overflow: 'hidden',
                  borderTop: `1px solid ${currentTheme.borderColor}`,
                  backgroundColor: currentTheme.surface
                }}
              >
                <div style={{ padding: '16px 20px' }}>
                  {/* Theme Toggle */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: '12px'
                  }}>
                    <span style={{
                      color: currentTheme.textPrimary,
                      fontSize: '14px',
                      fontWeight: '500'
                    }}>
                      Theme
                    </span>
                    <motion.button
                      onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      style={{
                        width: '50px',
                        height: '28px',
                        borderRadius: '14px',
                        backgroundColor: theme === 'dark' ? currentTheme.accentPrimary : currentTheme.borderColor,
                        border: 'none',
                        cursor: 'pointer',
                        position: 'relative',
                        transition: 'all 0.3s ease'
                      }}
                    >
                      <motion.div
                        animate={{ 
                          x: theme === 'dark' ? 22 : 0,
                        }}
                        transition={{ duration: 0.3, ease: "easeInOut" }}
                        style={{
                          position: 'absolute',
                          left: '2px',
                          top: '2px',
                          width: '24px',
                          height: '24px',
                          borderRadius: '12px',
                          backgroundColor: '#ffffff',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '12px',
                          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                        }}
                      >
                        {theme === 'light' ? '🌙' : '☀️'}
                      </motion.div>
                    </motion.button>
                  </div>

                  {/* Additional Settings can go here */}
                  <div style={{
                    fontSize: '12px',
                    color: currentTheme.textTertiary,
                    textAlign: 'center',
                    marginTop: '8px'
                  }}>
                    v1.0.0
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </motion.div>
  );
}

// Main App Component
function App() {
  const [activeChat, setActiveChat] = useState(1);
  const [chatSessions, setChatSessions] = useState(mockChatSessions);
  const [allChatMessages, setAllChatMessages] = useState(initialChatMessages);

  const handleChatSelect = (chatId) => {
    setActiveChat(chatId);
  };

  const handleNewChat = () => {
    const newChatId = Math.max(...chatSessions.map(chat => chat.id)) + 1;
    
    const newChat = {
      id: newChatId,
      title: `New Chat ${newChatId}`,
      lastMessage: "Start a conversation...",
      timestamp: "now",
      unread: 0
    };
    
    setAllChatMessages(prev => ({
      ...prev,
      [newChatId]: [{
        id: 1,
        type: "assistant",
        content: "Hello! I'm here to help you. What would you like to discuss?",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]
    }));
    
    setChatSessions(prev => [newChat, ...prev]);
    setActiveChat(newChatId);
  };

  const handleUpdateMessages = (chatId, messages) => {
    setAllChatMessages(prev => ({
      ...prev,
      [chatId]: messages
    }));
  };

  return (
    <ThemeProvider>
      <motion.div 
        layout
        style={{
          display: 'flex',
          height: '100vh',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          overflow: 'hidden'
        }}
      >
        <Sidebar
          activeChat={activeChat}
          chatMessages={allChatMessages}
          onUpdateMessages={handleUpdateMessages}
          chatSessions={chatSessions}
          onChatSelect={handleChatSelect}
          onNewChat={handleNewChat}
        />
        
        <motion.div 
          layout
          transition={{
            duration: 0.4,
            ease: [0.23, 1, 0.320, 1]
          }}
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            minHeight: 0,
            position: 'relative',
            willChange: 'width'
          }}
        >
          <ChatWindow
            activeChat={activeChat}
            chatMessages={allChatMessages}
            onUpdateMessages={handleUpdateMessages}
            chatSessions={chatSessions}
            onChatSelect={handleChatSelect}
          />
        </motion.div>
      </motion.div>
    </ThemeProvider>
  );
}

export default App;

import React, { useState } from 'react';
import axios from 'axios';



const RiskForm = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        setMessages(prev => [...prev, { text: input, isUser: true }]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await axios.post('/api/chat', { message: input });

            // Handle different response structures
            const analysis = response.data.patient?.analysis;
            if (analysis) {
                setMessages(prev => [
                    ...prev,
                    {
                        text: `Risk Level: ${analysis.risk_level}\nProbability: ${analysis.probability}%`,
                        isUser: false
                    }
                ]);
            } else {
                setMessages(prev => [
                    ...prev,
                    {
                        text: response.data.detail || "Analysis unavailable",
                        isUser: false
                    }
                ]);
            }

        } catch (error) {
            // Handle different error types
            const errorMessage = error.response?.data?.detail ||
                error.message ||
                "Analysis failed";

            setMessages(prev => [
                ...prev,
                {
                    text: `Error: ${errorMessage}`,
                    isUser: false,
                    isError: true
                }
            ]);
        } finally {
            setIsLoading(false);
        }
    };
    return (
        <div className="chat-container">
            <div className="messages">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.isUser ? 'user' : 'bot'}`}>
                        {msg.text.split('\n').map((line, i) => (
                            <p key={i}>{line}</p>
                        ))}
                    </div>
                ))}
            </div>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Enter patient info..."
                    disabled={isLoading}
                />
                <button type="submit" disabled={isLoading}>
                    {isLoading ? 'Analyzing...' : 'Send'}
                </button>
            </form>
        </div>
    );
};

export default RiskForm;
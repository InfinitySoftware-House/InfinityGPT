
class Templates:
    def Wikipedia():
        return """What is the subject of this sentence: {question}?
        The subject is:"""
    
    def Search():
        return """You are an assistant that makes DuckDuckGo search using the input
        
        examples:
        xxx xx xxxx
        xx xxx xx
        
        What should be a possible DuckDuckGo search for this question: {question}?
            
        search:"""
        
    def Generate():
        return """This is the result of a search: {result}
        Create an answer based on this question: {question}
        
        Write only one answer
        
        Answer:"""
        
    def Thinking():
        return """The action to take, it should be only one of this: {tools}
        
        Examples:
        Action: wikipedia
        Action: google
        Action: youtube
        
        Action: """
    
    def Chat():
        return """You are an helpful assistant, your name is {name}:
        
        Based on this user message: {message}
        And on this history: {history}
        Write an answer:"""
    
    def Code():
        return """You are an assistant that writes code based on user input
        This is the input: {input}
        Code:"""
        
    def Document():
        return """You are an assistant that answer an Input based on a Context:
        Input: {action}
        Context: {input}
        Result:"""
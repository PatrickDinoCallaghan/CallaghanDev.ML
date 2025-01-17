﻿namespace CallaghanDev.ML.Exceptions
{
 
    public class InfinityException : Exception
    {
        public InfinityException() : base("Infinity detected") { }

        public InfinityException(string message) : base(message) { }

        public InfinityException(string message, Exception inner) : base(message, inner) { }
    }

}
